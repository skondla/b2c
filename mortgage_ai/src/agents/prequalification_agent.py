"""Prequalification Agent — Stage 2 specialist (BRD §3.2, FR-010 through FR-014).

Uses a LCEL chain + ReAct reasoning to:
  1. Collect borrower's basic financial info
  2. Run soft credit check (no score impact)
  3. Evaluate eligibility across all loan products
  4. Generate pre-qualification letter reference
  5. Complete the entire flow in ≤ 60 seconds (NFR-002)

LangSmith traces every call automatically via LANGCHAIN_TRACING_V2=true.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.tools import (
    calculate_prequal,
    check_eligibility,
    search_mortgage_knowledge,
    soft_credit_check,
)

PREQUAL_SYSTEM = """You are an expert mortgage pre-qualification specialist for a B2C digital lending platform.

Your job is to help borrowers understand whether they pre-qualify for a mortgage quickly and clearly.
You must complete the pre-qualification in under 60 seconds of processing time.

When given borrower information, use your tools in this sequence:
1. soft_credit_check — retrieve credit score range (no score impact)
2. check_eligibility — determine eligible products and key ratios
3. calculate_prequal — generate pre-qualification with max loan amount and rate range

Format your final response as a friendly, clear summary that includes:
- Whether they pre-qualify (YES / CONDITIONAL / NO)
- Maximum loan amount
- Estimated rate range
- Recommended loan product
- DTI and LTV ratios
- Next steps

Use plain language (8th-grade reading level, per BRD §5.6).
Always include the disclaimer that this is not a commitment to lend.

If the borrower does NOT pre-qualify:
- Explain the specific reason(s) clearly and empathetically
- Provide 2-3 concrete steps they can take to improve their position
- Never be discouraging — frame it as "here's the path forward"
"""

PREQUAL_STRUCTURED_PROMPT = """Extract pre-qualification data from this borrower input and return ONLY valid JSON.

Borrower input: {user_input}

Return this exact JSON structure (use null for missing fields):
{{
  "borrower_name": "string or null",
  "annual_income": number or null,
  "monthly_debt_payments": number or null,
  "estimated_home_price": number or null,
  "down_payment": number or null,
  "loan_purpose": "purchase" or "refinance" or null,
  "property_type": "single_family" or "condo" or "multi_family_2_4" or "manufactured" or null,
  "veteran_status": true or false,
  "ssn_last4": "string or null",
  "date_of_birth": "YYYY-MM-DD or null"
}}
"""


def _build_llm():
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.primary_llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o", temperature=0)


class PrequalificationAgent:
    """Standalone pre-qualification agent for direct API invocation or testing."""

    def __init__(self):
        self._llm = _build_llm()
        self._tools = [soft_credit_check, calculate_prequal, check_eligibility, search_mortgage_knowledge]
        self._llm_with_tools = self._llm.bind_tools(self._tools)

    def extract_borrower_data(self, user_input: str) -> dict:
        """Parse unstructured borrower input into structured data."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You extract structured data from natural language. Return only valid JSON."),
            ("human", PREQUAL_STRUCTURED_PROMPT),
        ])
        chain = prompt | self._llm | JsonOutputParser()
        try:
            return chain.invoke({"user_input": user_input})
        except Exception:
            return {}

    def run(self, borrower_data: dict, conversation_history: list | None = None) -> dict:
        """Run the pre-qualification agent with structured borrower data.

        Args:
            borrower_data: Dict with annual_income, down_payment, etc.
            conversation_history: Optional prior messages for context

        Returns:
            Dict with prequal_result, messages, and updated state fields
        """
        messages = [SystemMessage(content=PREQUAL_SYSTEM)]
        if conversation_history:
            messages.extend(conversation_history)

        # Build context message for the LLM
        context = f"""Please pre-qualify this borrower:

Borrower: {borrower_data.get('borrower_name', 'Borrower')}
Annual Income: ${borrower_data.get('annual_income', 0):,.0f}
Monthly Debt Payments: ${borrower_data.get('monthly_debt_payments', 0):,.0f}
Estimated Home Price: ${borrower_data.get('estimated_home_price', 0):,.0f}
Down Payment: ${borrower_data.get('down_payment', 0):,.0f}
Loan Purpose: {borrower_data.get('loan_purpose', 'purchase')}
Property Type: {borrower_data.get('property_type', 'single_family')}
Veteran Status: {borrower_data.get('veteran_status', False)}

Please use the soft_credit_check, check_eligibility, and calculate_prequal tools to complete the assessment.
"""
        messages.append(HumanMessage(content=context))

        # ReAct loop: run LLM + tools until no more tool calls
        max_iterations = 5
        tool_map = {t.name: t for t in self._tools}
        result_messages = list(messages)

        for _ in range(max_iterations):
            response = self._llm_with_tools.invoke(result_messages)
            result_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            # Execute tool calls
            from langchain_core.messages import ToolMessage
            for tool_call in response.tool_calls:
                tool_fn = tool_map.get(tool_call["name"])
                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tool_call["args"])
                        result_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        ))
                    except Exception as e:
                        result_messages.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tool_call["id"],
                        ))

        # Extract final text response
        final_response = ""
        prequal_data = {}
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", []):
                final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Extract prequal data from tool results
        for msg in result_messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                try:
                    import json
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "eligible" in data:
                        prequal_data = data
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            "response": final_response,
            "prequal_data": prequal_data,
            "messages": result_messages,
            "prequal_completed_at": datetime.utcnow().isoformat(),
            "stage": "application" if prequal_data.get("eligible") else "prequalification",
        }

    def answer_question(self, question: str, prequal_context: dict | None = None) -> str:
        """Answer a one-off question using RAG over the mortgage knowledge base."""
        context_str = ""
        if prequal_context:
            context_str = f"\nBorrower context: max loan ${prequal_context.get('max_loan_amount', 0):,.0f}, "
            context_str += f"products: {', '.join(prequal_context.get('eligible_products', []))}"

        # Use knowledge search tool
        kb_results = search_mortgage_knowledge.invoke({
            "query": question,
            "n_results": 3,
        })

        context_content = "\n".join(
            r.get("content", "") for r in kb_results.get("results", [])
        )

        prompt = f"""Based on this mortgage knowledge:

{context_content}
{context_str}

Please answer this question clearly and in plain language:
{question}
"""
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content if isinstance(response.content, str) else str(response.content)
