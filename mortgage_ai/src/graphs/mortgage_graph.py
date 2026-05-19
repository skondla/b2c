"""B2C Mortgage Loan Application — LangGraph Workflow.

Implements all 9 borrower journey stages from BRD §3 as a stateful graph:
  Stage 1  Discovery & Account Creation
  Stage 2  Pre-Qualification       (credit check, eligibility engine)
  Stage 3  Full Application        (URLA Form 1003 guidance)
  Stage 4  Document Upload & Verification  (OCR, ML classification)
  Stage 5  Property & Loan Selection  (rate lock, payment calc)
  Stage 6  Disclosures & E-Signature  (TRID, E-SIGN)
  Stage 7  Underwriting & Conditions  (AUS, human review)
  Stage 8  Closing                 (RON, CD delivery)
  Stage 9  Post-Submission Self-Service (status, messaging)

LangSmith tracing enabled automatically via LANGCHAIN_TRACING_V2=true.
Human-in-the-loop checkpoints at underwriting and closing (interrupt_before).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.graphs.states import LoanStage, MortgageState, default_state
from src.tools import (
    calculate_apr,
    calculate_payment,
    capture_esign_consent,
    check_ecoa_compliance,
    check_eligibility,
    check_hmda_data,
    check_trid_compliance,
    classify_document,
    extract_document_data,
    get_current_rates,
    get_product_info,
    hard_credit_check,
    lock_rate,
    calculate_prequal,
    search_mortgage_knowledge,
    soft_credit_check,
    validate_document,
)

# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm(fast: bool = False):
    """Build the primary LLM with LangSmith tracing enabled automatically."""
    model = settings.fast_llm_model if fast else settings.primary_llm_model

    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
            temperature=0,
        )
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, temperature=0)

    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


# ── Tool groups per stage ──────────────────────────────────────────────────────

PREQUAL_TOOLS = [soft_credit_check, calculate_prequal, check_eligibility, search_mortgage_knowledge]
APPLICATION_TOOLS = [search_mortgage_knowledge, get_product_info, check_hmda_data]
DOCUMENT_TOOLS = [classify_document, extract_document_data, validate_document]
PROPERTY_TOOLS = [get_current_rates, calculate_payment, calculate_apr, lock_rate, get_product_info]
DISCLOSURE_TOOLS = [check_trid_compliance, capture_esign_consent, check_ecoa_compliance]
UNDERWRITING_TOOLS = [hard_credit_check, check_eligibility, check_ecoa_compliance, search_mortgage_knowledge]
CLOSING_TOOLS = [check_trid_compliance, search_mortgage_knowledge]
SELF_SERVICE_TOOLS = [search_mortgage_knowledge, get_product_info]

ALL_TOOLS = list({
    t.name: t for t in (
        PREQUAL_TOOLS + APPLICATION_TOOLS + DOCUMENT_TOOLS + PROPERTY_TOOLS
        + DISCLOSURE_TOOLS + UNDERWRITING_TOOLS + CLOSING_TOOLS + SELF_SERVICE_TOOLS
    )
}.values())


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    LoanStage.PREQUALIFICATION: """You are a friendly mortgage pre-qualification assistant for a B2C lending platform.
Your goal: help the borrower determine whether they pre-qualify for a mortgage in under 60 seconds (NFR-002).

Available tools: soft_credit_check, calculate_prequal, check_eligibility, search_mortgage_knowledge

Steps:
1. Confirm you have: estimated_home_price, down_payment, annual_income, monthly_debt_payments, credit_score_range_min, loan_purpose, property_type.
2. Call soft_credit_check with the borrower's details.
3. Call calculate_prequal with the financial inputs.
4. Return a clear, plain-language (8th-grade reading level) summary of eligibility, max loan, and rate range.
5. If ineligible, explain why and provide 2-3 actionable recommendations.

Always be encouraging, transparent about timelines and fees, and offer to answer questions.
Never promise a specific rate — always say "estimated" or "illustrative".
""",

    LoanStage.APPLICATION: """You are a mortgage application guide helping a borrower complete the URLA / Form 1003.
Guide them section by section. The required sections are:
  1. Borrower Information (personal details)
  2. Employment & Income
  3. Assets & Liabilities
  4. Property & Loan Information
  5. Declarations
  6. HMDA Demographic Information (optional — borrower may opt out)
  7. Loan Originator Information

After each section is complete, update urla_sections_completed in the state.
Call check_hmda_data when section 6 is reached. Use search_mortgage_knowledge for any borrower questions.
When all sections are complete, set urla_complete=True and advance to DOCUMENT_UPLOAD stage.
Save progress on every field change (FR-022). Plain language (8th-grade level).
""",

    LoanStage.DOCUMENT_UPLOAD: """You are a document collection specialist for a mortgage application.
Required documents for this application:
  - 2 most recent pay stubs (or 2 years tax returns if self-employed)
  - Last 2 W-2 forms
  - Last 2 months bank statements
  - Government-issued photo ID

For each uploaded document:
1. Call classify_document to identify the document type.
2. Call extract_document_data to pull key financial figures.
3. Call validate_document to check completeness and income consistency.
4. If valid: add to documents_verified list.
5. If invalid: explain the problem clearly and request resubmission.

When all required documents are verified, advance to PROPERTY_LOAN_SELECTION.
If a document fails repeatedly, flag for human review.
""",

    LoanStage.PROPERTY_LOAN_SELECTION: """You are a mortgage product advisor helping the borrower select the best loan.
Steps:
1. Confirm property_address is captured.
2. Call get_current_rates to show available products for this borrower's profile.
3. Walk the borrower through the top 2-3 product options with plain-language pros/cons.
4. Call calculate_payment for the selected product to show full PITI breakdown.
5. Call calculate_apr to compute APR for TRID disclosure.
6. Offer rate lock (15/30/45/60 days) — call lock_rate when borrower confirms.
7. Advance to DISCLOSURES stage once rate is locked or borrower declines lock.
""",

    LoanStage.DISCLOSURES: """You are a compliance specialist delivering required mortgage disclosures.
Required actions:
1. First: capture e-consent via capture_esign_consent before delivering ANY documents (FR-041, E-SIGN Act).
2. Verify TRID timing for Loan Estimate: call check_trid_compliance (must be within 3 business days of application).
3. Deliver Loan Estimate — confirm borrower has reviewed and signed (FR-040, FR-042).
4. Deliver Intent to Proceed — collect signature.
5. Log all delivery timestamps and IP addresses for audit trail (FR-043).
Advance to UNDERWRITING once all required disclosures are signed.
Never skip e-consent — this is a legal requirement.
""",

    LoanStage.UNDERWRITING: """You are an underwriting coordinator managing the loan file through automated and human underwriting.
Steps:
1. Confirm hard_pull_consented is True. If not, obtain consent first.
2. Call hard_credit_check to get the full FICO score and report ID.
3. Call check_eligibility with the final loan parameters and FICO score.
4. Submit to Automated Underwriting System (AUS) — simulate decision.
5. If Approve/Eligible: set conditional approval timestamp, list conditions.
6. If Refer: flag for human underwriter review.
7. If Refer with Caution or Decline: call check_ecoa_compliance to generate adverse action notice.
Update conditions list and advance to CLOSING when all conditions are cleared.
""",

    LoanStage.CLOSING: """You are a closing coordinator managing the final steps.
Steps:
1. Confirm closing_date is set.
2. Verify Closing Disclosure timing: call check_trid_compliance (must be 3 business days before closing).
3. Deliver Closing Disclosure — borrower must acknowledge receipt.
4. If RON is available (ron_available=True): initiate remote online notarization session.
5. Coordinate with title and settlement systems.
6. Once closing is complete: set closed_at timestamp and advance to POST_SUBMISSION.
""",

    LoanStage.POST_SUBMISSION: """You are a post-closing self-service assistant.
Help the borrower with:
- Checking their application status
- Understanding what their next steps are
- Answering questions about their loan
- Downloading documents from their vault
- Scheduling calls with their loan officer
Use search_mortgage_knowledge for any questions. Be proactive about upcoming deadlines.
""",
}


# ── Node functions ─────────────────────────────────────────────────────────────

def _make_agent_node(stage: LoanStage, tools: list, fast: bool = False):
    """Factory: create a lazily-initialized ReAct-style agent node for a given stage."""
    system_prompt = SYSTEM_PROMPTS.get(stage, "You are a helpful mortgage assistant.")
    _cache: dict = {}  # lazy cache for LLM instance

    def agent_node(state: dict) -> dict:
        if "llm" not in _cache:
            llm = _build_llm(fast=fast)
            _cache["llm"] = llm.bind_tools(tools)
        messages = state.get("messages", [])
        sys_msg = SystemMessage(content=system_prompt)
        response = _cache["llm"].invoke([sys_msg] + messages)
        return {"messages": [response]}

    agent_node.__name__ = f"{stage.value}_agent"
    return agent_node


def supervisor_node(state: dict) -> dict:
    """Route borrower to the correct stage based on state."""
    # Supervisor just passes through — routing is handled by conditional edges
    return {}


def discovery_node(state: dict) -> dict:
    """Stage 1: Welcome message and account confirmation (BRD §3.1)."""
    welcome = (
        f"Welcome to your mortgage application, {state.get('borrower_name', 'there')}! "
        f"I'm your AI mortgage assistant. I'll guide you through every step.\n\n"
        f"Here's what we'll cover:\n"
        f"1. Pre-qualification (takes ~2 minutes)\n"
        f"2. Full application (URLA Form 1003, ~20 minutes)\n"
        f"3. Document upload (pay stubs, W-2s, bank statements)\n"
        f"4. Property and loan selection\n"
        f"5. Disclosures and e-signature\n"
        f"6. Underwriting\n"
        f"7. Closing\n\n"
        f"Your progress is saved automatically — you can pause and resume at any time.\n"
        f"Ready to get started? Let's begin with your pre-qualification."
    )
    return {
        "messages": [AIMessage(content=welcome)],
        "stage": LoanStage.PREQUALIFICATION.value,
        "next_action": "proceed_to_prequal",
    }


def decline_node(state: dict) -> dict:
    """Generate ECOA-compliant adverse action notice."""
    message = (
        "I'm sorry — based on the information provided, we are unable to approve your mortgage "
        "application at this time.\n\n"
        "You will receive an Adverse Action Notice explaining the specific reasons. "
        "This is required by the Equal Credit Opportunity Act (ECOA).\n\n"
        "You have the right to:\n"
        "• Request a copy of your credit report\n"
        "• Dispute inaccurate information\n"
        "• Reapply after addressing the noted issues\n\n"
        "Our loan officers are available to discuss your options and help you prepare for the future."
    )
    return {
        "messages": [AIMessage(content=message)],
        "stage": LoanStage.DECLINED.value,
        "compliance_flags": state.get("compliance_flags", []) + ["adverse_action_notice_required"],
    }


def human_review_node(state: dict) -> dict:
    """Checkpoint: pause for human loan officer review."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "Your file has been flagged for review by one of our loan officers. "
                    "This is a normal part of the process for complex applications. "
                    "You'll receive an update within 1 business day."
                )
            )
        ],
        "human_review_required": True,
        "next_action": "awaiting_human_review",
    }


# ── Stage-specific agent nodes ──────────────────────────────────────────────────

prequal_agent = _make_agent_node(LoanStage.PREQUALIFICATION, PREQUAL_TOOLS)
application_agent = _make_agent_node(LoanStage.APPLICATION, APPLICATION_TOOLS)
document_agent = _make_agent_node(LoanStage.DOCUMENT_UPLOAD, DOCUMENT_TOOLS)
property_agent = _make_agent_node(LoanStage.PROPERTY_LOAN_SELECTION, PROPERTY_TOOLS)
disclosure_agent = _make_agent_node(LoanStage.DISCLOSURES, DISCLOSURE_TOOLS)
underwriting_agent = _make_agent_node(LoanStage.UNDERWRITING, UNDERWRITING_TOOLS)
closing_agent = _make_agent_node(LoanStage.CLOSING, CLOSING_TOOLS)
self_service_agent = _make_agent_node(LoanStage.POST_SUBMISSION, SELF_SERVICE_TOOLS, fast=True)


# ── Routing functions ─────────────────────────────────────────────────────────

def route_by_stage(state: dict) -> str:
    """Supervisor: route to the node matching the current stage."""
    stage = state.get("stage", LoanStage.DISCOVERY.value)
    route_map = {
        LoanStage.DISCOVERY.value: "discovery",
        LoanStage.PREQUALIFICATION.value: "prequalification",
        LoanStage.APPLICATION.value: "application",
        LoanStage.DOCUMENT_UPLOAD.value: "document_upload",
        LoanStage.PROPERTY_LOAN_SELECTION.value: "property_loan_selection",
        LoanStage.DISCLOSURES.value: "disclosures",
        LoanStage.UNDERWRITING.value: "underwriting",
        LoanStage.CLOSING.value: "closing",
        LoanStage.POST_SUBMISSION.value: "post_submission",
        LoanStage.COMPLETED.value: END,
        LoanStage.DECLINED.value: "decline",
        LoanStage.ON_HOLD.value: "human_review",
    }
    return route_map.get(stage, "discovery")


def should_use_tools(state: dict) -> Literal["tools", "supervisor"]:
    """Check if the last AI message has tool calls pending."""
    messages = state.get("messages", [])
    if not messages:
        return "supervisor"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "supervisor"


def after_prequal(state: dict) -> Literal["prequal_tools", "supervisor", "decline"]:
    """After pre-qual agent runs, route based on tool calls or eligibility."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "prequal_tools"
    if state.get("prequal_eligible") is False:
        return "decline"
    return "supervisor"


def after_underwriting(state: dict) -> Literal["underwriting_tools", "human_review", "supervisor", "decline"]:
    """After underwriting agent runs."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "underwriting_tools"
    decision = state.get("aus_decision", "")
    if decision == "Refer with Caution" or state.get("human_review_required"):
        return "human_review"
    if decision and "decline" in decision.lower():
        return "decline"
    return "supervisor"


def generic_tool_router(state: dict) -> Literal["tools", "supervisor"]:
    return should_use_tools(state)


# ── Graph construction ────────────────────────────────────────────────────────

def create_mortgage_graph():
    """Build and compile the B2C mortgage LangGraph workflow.

    Graph topology:
      START -> supervisor -> [stage nodes] -> tools -> supervisor -> ...
      With human-in-the-loop interrupts at: underwriting, closing
    """
    builder = StateGraph(dict)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("discovery", discovery_node)
    builder.add_node("prequalification", prequal_agent)
    builder.add_node("application", application_agent)
    builder.add_node("document_upload", document_agent)
    builder.add_node("property_loan_selection", property_agent)
    builder.add_node("disclosures", disclosure_agent)
    builder.add_node("underwriting", underwriting_agent)
    builder.add_node("closing", closing_agent)
    builder.add_node("post_submission", self_service_agent)
    builder.add_node("human_review", human_review_node)
    builder.add_node("decline", decline_node)

    # Shared ToolNode for all tools
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    # Stage-specific tool nodes (for clean separation of concerns)
    builder.add_node("prequal_tools", ToolNode(PREQUAL_TOOLS))
    builder.add_node("underwriting_tools", ToolNode(UNDERWRITING_TOOLS))

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Entry: START -> supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor routes to current stage
    builder.add_conditional_edges(
        "supervisor",
        route_by_stage,
        {
            "discovery": "discovery",
            "prequalification": "prequalification",
            "application": "application",
            "document_upload": "document_upload",
            "property_loan_selection": "property_loan_selection",
            "disclosures": "disclosures",
            "underwriting": "underwriting",
            "closing": "closing",
            "post_submission": "post_submission",
            "human_review": "human_review",
            "decline": "decline",
            END: END,
        },
    )

    # Discovery always advances to supervisor (which routes to prequal)
    builder.add_edge("discovery", "supervisor")

    # Pre-qualification: agent -> tools -> back to supervisor
    builder.add_conditional_edges(
        "prequalification",
        after_prequal,
        {
            "prequal_tools": "prequal_tools",
            "decline": "decline",
            "supervisor": "supervisor",
        },
    )
    builder.add_edge("prequal_tools", "supervisor")

    # Application: agent -> tools -> supervisor
    builder.add_conditional_edges("application", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Document upload: agent -> tools -> supervisor
    builder.add_conditional_edges("document_upload", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Property/loan selection: agent -> tools -> supervisor
    builder.add_conditional_edges("property_loan_selection", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Disclosures: agent -> tools -> supervisor
    builder.add_conditional_edges("disclosures", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Underwriting: agent -> tools or human review or decline
    builder.add_conditional_edges(
        "underwriting",
        after_underwriting,
        {
            "underwriting_tools": "underwriting_tools",
            "human_review": "human_review",
            "supervisor": "supervisor",
            "decline": "decline",
        },
    )
    builder.add_edge("underwriting_tools", "supervisor")

    # Closing: agent -> tools -> supervisor
    builder.add_conditional_edges("closing", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Post-submission: agent -> tools -> supervisor (loops)
    builder.add_conditional_edges("post_submission", generic_tool_router, {"tools": "tools", "supervisor": "supervisor"})

    # Shared tools always return to supervisor
    builder.add_edge("tools", "supervisor")

    # Human review -> supervisor (after loan officer acts)
    builder.add_edge("human_review", "supervisor")

    # Terminal nodes
    builder.add_edge("decline", END)

    # Compile with:
    # - MemorySaver for cross-device persistence (FR-014, FR-022)
    # - interrupt_before underwriting and closing for human-in-the-loop
    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # Pause for LO review before proceeding
    )
    return graph


# ── Public API ────────────────────────────────────────────────────────────────

_graph_instance = None


def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_mortgage_graph()
    return _graph_instance


def run_mortgage_workflow(
    state: dict,
    user_message: str,
    thread_id: str,
) -> tuple[dict, str]:
    """Run one turn of the mortgage workflow.

    Args:
        state: Current MortgageState dict
        user_message: Borrower's input text
        thread_id: Application ID used as LangGraph thread ID for persistence

    Returns:
        (updated_state, assistant_response_text)
    """
    graph = get_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    # Inject user message into state
    input_state = dict(state)
    input_state["messages"] = state.get("messages", []) + [HumanMessage(content=user_message)]

    result = graph.invoke(input_state, config=config)

    # Extract the last AI message as the response
    response_text = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return result, response_text


def stream_mortgage_workflow(
    state: dict,
    user_message: str,
    thread_id: str,
):
    """Streaming version — yields (event_type, data) tuples for SSE."""
    graph = get_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    input_state = dict(state)
    input_state["messages"] = state.get("messages", []) + [HumanMessage(content=user_message)]

    for event in graph.stream(input_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                yield "end", {}
                continue
            messages = node_output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    yield "message", {"node": node_name, "content": content}
            # Yield stage transitions
            if "stage" in node_output:
                yield "stage_change", {"stage": node_output["stage"]}
