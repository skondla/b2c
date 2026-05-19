"""Loan Officer Copilot — BRD §2.1 Persona 3 ("Sam"), FR-024, FR-052, FR-053.

Provides loan officers with:
  • Full application state visibility
  • AI-assisted analysis and recommendations
  • Co-browse capability (view/edit on behalf of borrower with consent)
  • Pipeline exception handling
  • Condition clearing assistant
  • Regulatory decision support (adverse action, HMDA)

LangSmith traces every LO action for compliance audit trail (FR-043).
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import settings
from src.tools import (
    calculate_payment,
    check_ecoa_compliance,
    check_eligibility,
    check_trid_compliance,
    get_current_rates,
    search_mortgage_knowledge,
    validate_document,
)


LOAN_OFFICER_SYSTEM = """You are an AI copilot assisting a licensed mortgage loan officer.

Your role is to help the loan officer:
1. Quickly understand any borrower's application status and risk profile
2. Identify missing information, compliance issues, or exceptions that need attention
3. Draft communications to borrowers (always in plain, 8th-grade language)
4. Ensure regulatory compliance (TRID timing, ECOA, HMDA)
5. Suggest solutions for common underwriting conditions
6. Calculate scenarios (payment, DTI, LTV) for rate advisory conversations

IMPORTANT CONSTRAINTS:
- Any changes to the borrower's application require explicit borrower consent (FR-024)
- Never take adverse action without checking ECOA compliance first
- All recommendations must be documented for audit trail compliance
- TRID disclosure deadlines are non-negotiable — flag any approaching deadlines immediately

You have access to tools: check_eligibility, get_current_rates, calculate_payment,
check_trid_compliance, check_ecoa_compliance, search_mortgage_knowledge, validate_document.

When analyzing a file:
1. Summarize the current stage and any red flags
2. List any compliance concerns with urgency
3. Provide specific, actionable next steps
"""

BORROWER_EMAIL_SYSTEM = """You are drafting a professional, friendly email to a mortgage borrower.
Requirements:
- Plain language (8th-grade reading level)
- No technical jargon without explanation
- Clear call-to-action
- Empathetic and encouraging tone
- Include loan officer contact information
- Under 200 words
"""


def _build_llm():
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.primary_llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.1,  # Slight creativity for communications
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o", temperature=0.1)


class LoanOfficerCopilot:
    """AI copilot for loan officers — pipeline visibility and decision support."""

    def __init__(self):
        self._llm = _build_llm()
        self._tools = [
            check_eligibility, get_current_rates, calculate_payment,
            check_trid_compliance, check_ecoa_compliance,
            search_mortgage_knowledge, validate_document,
        ]
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._tool_map = {t.name: t for t in self._tools}

    def analyze_application(self, state: dict, lo_question: str = "") -> str:
        """Provide LO with an AI-powered analysis of the current application."""
        stage = state.get("stage", "unknown")
        borrower = state.get("borrower_name", "Borrower")
        income = state.get("annual_income", 0)
        loan_amount = state.get("loan_amount", 0)
        home_price = state.get("estimated_home_price", 0)
        down_payment = state.get("down_payment", 0)
        credit_score = state.get("credit_score")
        aus_decision = state.get("aus_decision", "pending")
        conditions = state.get("conditions", [])
        compliance_flags = state.get("compliance_flags", [])
        missing_docs = state.get("missing_documents", [])

        analysis_prompt = f"""Analyze this mortgage application file for the loan officer:

APPLICATION SUMMARY:
- Borrower: {borrower}
- Stage: {stage}
- Annual Income: ${income:,.0f}
- Loan Amount: ${loan_amount:,.0f}
- Home Price: ${home_price:,.0f}
- Down Payment: ${down_payment:,.0f}
- LTV: {(loan_amount/home_price*100 if home_price else 0):.1f}%
- Credit Score: {credit_score or 'Not yet pulled'}
- AUS Decision: {aus_decision}
- Open Conditions: {len(conditions)} ({', '.join(conditions[:3]) if conditions else 'None'})
- Missing Documents: {', '.join(missing_docs) if missing_docs else 'None'}
- Compliance Flags: {', '.join(compliance_flags) if compliance_flags else 'None'}

{f"Loan Officer Question: {lo_question}" if lo_question else "Please provide a comprehensive file review."}

Provide:
1. File health score (1-10) and brief explanation
2. Top 3 priority items requiring action
3. Any compliance concerns with urgency level
4. Recommended next steps
"""
        messages = [
            SystemMessage(content=LOAN_OFFICER_SYSTEM),
            HumanMessage(content=analysis_prompt),
        ]

        response = self._llm_with_tools.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    def draft_borrower_communication(
        self,
        communication_type: str,
        borrower_name: str,
        context: dict,
        lo_name: str = "Your Loan Officer",
        lo_email: str = "loans@mortgage.example.com",
        lo_phone: str = "1-800-MORTGAGE",
    ) -> str:
        """Draft a borrower-facing email or message.

        communication_type: 'condition_request', 'status_update',
                            'document_request', 'approval_notification',
                            'scheduling_closing', 'adverse_action'
        """
        templates = {
            "condition_request": (
                f"Draft a professional email to {borrower_name} requesting the following "
                f"underwriting conditions to be cleared: {context.get('conditions', [])}. "
                f"Explain why each item is needed in plain terms."
            ),
            "status_update": (
                f"Draft a status update email to {borrower_name} about their mortgage application. "
                f"Current stage: {context.get('stage', 'in progress')}. "
                f"Key update: {context.get('update_text', 'Your application is progressing well.')}. "
                f"Next step for the borrower: {context.get('next_step', 'We will contact you soon.')}."
            ),
            "document_request": (
                f"Draft a document request email to {borrower_name}. "
                f"Needed documents: {context.get('missing_docs', [])}. "
                f"Explain how to submit them digitally."
            ),
            "approval_notification": (
                f"Draft a conditional approval congratulations email to {borrower_name}. "
                f"Loan amount: ${context.get('loan_amount', 0):,.0f}. "
                f"Rate: {context.get('interest_rate', 0):.3f}%. "
                f"Remaining conditions: {context.get('conditions', [])}."
            ),
            "scheduling_closing": (
                f"Draft a closing scheduling email to {borrower_name}. "
                f"Proposed closing date: {context.get('closing_date', 'TBD')}. "
                f"Include instructions about what to bring and what to expect."
            ),
            "adverse_action": (
                f"Draft an adverse action notice for {borrower_name}. "
                f"Reason: {context.get('decline_reason', 'Application was unable to be approved')}. "
                f"Must include ECOA rights statement. This is a legal document — be precise."
            ),
        }

        prompt_text = templates.get(
            communication_type,
            f"Draft a professional mortgage communication to {borrower_name} about: {context}"
        )

        messages = [
            SystemMessage(content=BORROWER_EMAIL_SYSTEM),
            HumanMessage(content=f"""{prompt_text}

Loan officer: {lo_name}
Contact: {lo_email} | {lo_phone}

Write the full email including subject line."""),
        ]

        response = self._llm.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    def calculate_scenarios(
        self,
        loan_amount: float,
        home_value: float,
        credit_score: int,
        income: float,
        monthly_debts: float,
    ) -> str:
        """Generate rate/payment scenarios for LO advisory conversation."""
        # Get current rates
        rates_result = get_current_rates.invoke({
            "loan_purpose": "purchase",
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "home_value": home_value,
        })

        products = rates_result.get("available_products", [])[:3]
        monthly_income = income / 12

        scenarios = []
        for product in products:
            pmt = product.get("monthly_payment_estimate", 0)
            dti = (pmt + monthly_debts) / monthly_income if monthly_income > 0 else 0
            scenarios.append(
                f"• {product['product']}: {product['interest_rate']:.3f}% → "
                f"${pmt:,.0f}/mo P&I | DTI: {dti:.1%}"
            )

        return (
            f"Loan Scenarios for ${loan_amount:,.0f} loan (FICO {credit_score}):\n\n"
            + "\n".join(scenarios)
            + f"\n\nMonthly Income: ${monthly_income:,.0f}"
            + f"\nExisting Monthly Debts: ${monthly_debts:,.0f}"
        )

    def check_compliance_calendar(self, state: dict) -> list[dict]:
        """Return list of upcoming compliance deadlines for this application."""
        alerts = []
        created_at = state.get("created_at", datetime.utcnow().isoformat())
        disclosures = state.get("disclosures_delivered", [])
        closing_date = state.get("closing_date")

        # Loan Estimate: must be delivered within 3 business days of application
        if "loan_estimate" not in disclosures:
            alerts.append({
                "type": "TRID_LOAN_ESTIMATE",
                "urgency": "HIGH",
                "message": "Loan Estimate not yet delivered — required within 3 business days of application",
                "regulation": "12 CFR § 1026.19(e)",
            })

        # Closing Disclosure: 3 business days before closing
        if closing_date and "closing_disclosure" not in disclosures:
            alerts.append({
                "type": "TRID_CLOSING_DISCLOSURE",
                "urgency": "CRITICAL",
                "message": f"Closing Disclosure must be delivered 3 business days before {closing_date}",
                "regulation": "12 CFR § 1026.19(f)",
            })

        # ECOA: 30 days for adverse action
        if state.get("stage") == "declined":
            alerts.append({
                "type": "ECOA_ADVERSE_ACTION",
                "urgency": "CRITICAL",
                "message": "Adverse Action Notice must be sent within 30 days of application receipt",
                "regulation": "12 CFR § 1002.9",
            })

        # HMDA reporting
        if not state.get("hmda_complete"):
            alerts.append({
                "type": "HMDA_INCOMPLETE",
                "urgency": "MEDIUM",
                "message": "HMDA demographic data collection incomplete",
                "regulation": "12 CFR § 1003.4",
            })

        return alerts

    def answer_regulatory_question(self, question: str) -> str:
        """Answer LO regulatory/compliance questions using knowledge base RAG."""
        kb_results = search_mortgage_knowledge.invoke({
            "query": question,
            "n_results": 4,
            "filter_category": "regulations",
        })
        context = "\n".join(r.get("content", "") for r in kb_results.get("results", []))

        messages = [
            SystemMessage(content=(
                "You are an expert mortgage compliance advisor. Answer the loan officer's "
                "regulatory question accurately and cite the relevant regulation."
            )),
            HumanMessage(content=f"Knowledge base context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)
