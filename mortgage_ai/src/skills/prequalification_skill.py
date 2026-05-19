"""Pre-qualification Skill — composable LCEL chain for fast eligibility checks.

Designed to complete within 60 seconds (NFR-002) as a standalone skill
callable from the graph, API, or Loan Officer copilot.
"""

from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import settings
from src.tools import calculate_prequal, check_eligibility, soft_credit_check


def _build_llm():
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.fast_llm_model, api_key=settings.anthropic_api_key, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You generate clear, friendly pre-qualification summaries for mortgage borrowers.
Return ONLY valid JSON with this exact structure:
{{
  "headline": "one-line result (e.g., 'You pre-qualify for up to $450,000!')",
  "eligible": true or false,
  "max_loan_formatted": "$XXX,XXX",
  "rate_range": "X.XXX% - X.XXX%",
  "recommended_product_display": "30-Year Fixed Conventional",
  "monthly_payment_estimate": "$X,XXX/month (principal & interest)",
  "dti_ratio": "XX.X%",
  "ltv_ratio": "XX.X%",
  "next_steps": ["step 1", "step 2", "step 3"],
  "ineligibility_reasons": [] or ["reason1"],
  "recommendations": [] or ["action to improve"],
  "disclaimer": "This pre-qualification is not a commitment to lend..."
}}""",
    ),
    ("human", "Pre-qualification data: {prequal_data}\n\nEligibility data: {eligibility_data}"),
])


class PrequalificationSkill:
    """Composable skill: runs all prequal tools and returns a structured summary."""

    def __init__(self):
        self._llm = _build_llm()
        self._summary_chain = SUMMARY_PROMPT | self._llm | JsonOutputParser()

    def run(
        self,
        annual_income: float,
        monthly_debt_payments: float,
        estimated_home_price: float,
        down_payment: float,
        credit_score_range_min: int,
        loan_purpose: str = "purchase",
        property_type: str = "single_family",
        veteran_status: bool = False,
        borrower_name: str = "Borrower",
        ssn_last4: str = "0000",
        date_of_birth: str = "1990-01-01",
    ) -> dict:
        """Run the full pre-qualification skill pipeline.

        Returns:
            Structured pre-qualification result dict
        """
        # Step 1: Soft credit check
        credit_result = soft_credit_check.invoke({
            "borrower_name": borrower_name,
            "ssn_last4": ssn_last4,
            "annual_income": annual_income,
            "date_of_birth": date_of_birth,
        })

        # Use actual simulated score for eligibility (not just range min)
        actual_score_min = credit_result.get("estimated_score", credit_score_range_min)

        # Step 2: Check eligibility
        loan_amount = estimated_home_price - down_payment
        eligibility_result = check_eligibility.invoke({
            "estimated_home_price": estimated_home_price,
            "down_payment": down_payment,
            "annual_income": annual_income,
            "monthly_debt_payments": monthly_debt_payments,
            "credit_score": actual_score_min,
            "loan_purpose": loan_purpose,
            "property_type": property_type,
            "veteran_status": veteran_status,
        })

        # Step 3: Calculate pre-qual (generates letter reference)
        prequal_result = calculate_prequal.invoke({
            "estimated_home_price": estimated_home_price,
            "down_payment": down_payment,
            "annual_income": annual_income,
            "monthly_debt_payments": monthly_debt_payments,
            "credit_score_range_min": actual_score_min,
            "loan_purpose": loan_purpose,
            "property_type": property_type,
            "veteran_status": veteran_status,
        })

        # Step 4: Generate human-readable summary
        try:
            summary = self._summary_chain.invoke({
                "prequal_data": str(prequal_result),
                "eligibility_data": str(eligibility_result),
            })
        except Exception:
            # Fallback if LLM is unavailable
            summary = {
                "headline": (
                    f"You pre-qualify for up to ${prequal_result.get('max_loan_amount', loan_amount):,.0f}!"
                    if prequal_result.get("eligible")
                    else "Pre-qualification could not be completed with current information."
                ),
                "eligible": prequal_result.get("eligible", False),
                "max_loan_formatted": f"${prequal_result.get('max_loan_amount', loan_amount):,.0f}",
                "rate_range": f"{prequal_result.get('estimated_rate_min', 0):.3f}% - {prequal_result.get('estimated_rate_max', 0):.3f}%",
                "recommended_product_display": prequal_result.get("recommended_product", ""),
                "monthly_payment_estimate": f"${prequal_result.get('estimated_monthly_payment', 0):,.0f}/month",
                "dti_ratio": f"{prequal_result.get('dti_ratio', 0):.1%}",
                "ltv_ratio": f"{prequal_result.get('ltv_ratio', 0):.1%}",
                "next_steps": ["Complete your full application", "Upload required documents"],
                "ineligibility_reasons": eligibility_result.get("ineligibility_reasons", []),
                "recommendations": [],
                "disclaimer": prequal_result.get("disclosure", "Not a commitment to lend."),
            }

        return {
            **summary,
            "raw_prequal": prequal_result,
            "raw_eligibility": eligibility_result,
            "raw_credit": credit_result,
            "prequal_letter_url": prequal_result.get("letter_url"),
            "prequal_letter_id": prequal_result.get("prequal_letter_id"),
            "eligible_products": prequal_result.get("eligible_products", []),
            "credit_score_range": credit_result.get("score_range", ""),
            "credit_inquiry_id": credit_result.get("inquiry_id"),
        }
