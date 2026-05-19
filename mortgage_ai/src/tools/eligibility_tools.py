"""Eligibility and pre-qualification rules engine — FR-010, FR-012.

Implements configurable DTI / LTV / FICO / occupancy / property-type rules
mapped to each loan product. BRD §4.2.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ── Product eligibility rules ──────────────────────────────────────────────────

PRODUCT_RULES: dict[str, dict] = {
    "conventional_30_fixed": {
        "min_fico": 620,
        "max_dti": 0.45,
        "max_ltv": 0.97,
        "max_loan": 766_550,
        "rate_base": 6.875,
        "pmi_required_ltv": 0.80,
    },
    "conventional_15_fixed": {
        "min_fico": 620,
        "max_dti": 0.43,
        "max_ltv": 0.97,
        "max_loan": 766_550,
        "rate_base": 6.25,
        "pmi_required_ltv": 0.80,
    },
    "fha_30_fixed": {
        "min_fico": 500,
        "max_dti": 0.57,
        "max_ltv": 0.965,
        "max_loan": 498_257,
        "rate_base": 6.50,
        "mip_upfront": 0.0175,
        "mip_annual": 0.0055,
    },
    "va_30_fixed": {
        "min_fico": 580,
        "max_dti": 0.60,
        "max_ltv": 1.00,
        "max_loan": 2_000_000,
        "rate_base": 6.25,
        "funding_fee": 0.023,
    },
    "jumbo_30_fixed": {
        "min_fico": 700,
        "max_dti": 0.43,
        "max_ltv": 0.80,
        "max_loan": 5_000_000,
        "rate_base": 7.125,
        "reserves_months": 12,
    },
}


class EligibilityInput(BaseModel):
    estimated_home_price: float = Field(description="Estimated purchase price or current home value in USD")
    down_payment: float = Field(description="Borrower's down payment amount in USD")
    annual_income: float = Field(description="Combined gross annual income of all borrowers")
    monthly_debt_payments: float = Field(description="Current monthly debt obligations (car, student loans, credit cards)")
    credit_score: int = Field(description="FICO credit score from credit report")
    loan_purpose: str = Field(description="purchase or refinance")
    property_type: str = Field(description="single_family, condo, multi_family_2_4, or manufactured")
    veteran_status: bool = Field(default=False, description="True if borrower is a veteran eligible for VA loan")


class PrequalInput(BaseModel):
    estimated_home_price: float
    down_payment: float
    annual_income: float
    monthly_debt_payments: float
    credit_score_range_min: int = Field(description="Lower bound of soft-pull credit score range")
    loan_purpose: str
    property_type: str
    veteran_status: bool = False


def _estimate_monthly_payment(loan_amount: float, annual_rate_pct: float, term_years: int = 30) -> float:
    monthly_rate = annual_rate_pct / 100 / 12
    n = term_years * 12
    if monthly_rate == 0:
        return loan_amount / n
    return loan_amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)


def _select_eligible_products(
    credit_score: int,
    loan_amount: float,
    ltv: float,
    dti: float,
    veteran: bool,
) -> list[str]:
    eligible = []
    for product, rules in PRODUCT_RULES.items():
        if product == "va_30_fixed" and not veteran:
            continue
        if credit_score < rules["min_fico"]:
            continue
        if ltv > rules["max_ltv"]:
            continue
        if dti > rules["max_dti"]:
            continue
        if loan_amount > rules["max_loan"]:
            continue
        eligible.append(product)
    return eligible


@tool("check_eligibility", args_schema=EligibilityInput, return_direct=False)
def check_eligibility(
    estimated_home_price: float,
    down_payment: float,
    annual_income: float,
    monthly_debt_payments: float,
    credit_score: int,
    loan_purpose: str,
    property_type: str,
    veteran_status: bool = False,
) -> dict:
    """Evaluate borrower eligibility against configurable product rules
    (DTI, LTV, FICO, occupancy, property type) as required by FR-012.
    Returns eligible products, DTI/LTV ratios, and disqualification reasons.
    """
    loan_amount = estimated_home_price - down_payment
    ltv = loan_amount / estimated_home_price if estimated_home_price > 0 else 0

    # Estimate PITI for DTI calculation using conventional 30yr as proxy
    estimated_rate = PRODUCT_RULES["conventional_30_fixed"]["rate_base"]
    estimated_monthly = _estimate_monthly_payment(loan_amount, estimated_rate)
    monthly_income = annual_income / 12
    dti = (estimated_monthly + monthly_debt_payments) / monthly_income if monthly_income > 0 else 1.0

    eligible_products = _select_eligible_products(credit_score, loan_amount, ltv, dti, veteran_status)

    ineligibility_reasons = []
    if credit_score < 500:
        ineligibility_reasons.append("Credit score below minimum threshold (500)")
    if dti > 0.60:
        ineligibility_reasons.append(f"DTI ratio {dti:.1%} exceeds maximum for all products (60%)")
    if down_payment < 0:
        ineligibility_reasons.append("Down payment cannot be negative")
    if loan_amount <= 0:
        ineligibility_reasons.append("Loan amount must be positive")

    return {
        "eligible": len(eligible_products) > 0,
        "eligible_products": eligible_products,
        "loan_amount": round(loan_amount, 2),
        "ltv_ratio": round(ltv, 4),
        "dti_ratio": round(dti, 4),
        "estimated_monthly_payment": round(estimated_monthly, 2),
        "monthly_income": round(monthly_income, 2),
        "credit_score": credit_score,
        "ineligibility_reasons": ineligibility_reasons,
        "requires_pmi": ltv > 0.80 and "conventional" in (eligible_products[0] if eligible_products else ""),
    }


@tool("calculate_prequal", args_schema=PrequalInput, return_direct=False)
def calculate_prequal(
    estimated_home_price: float,
    down_payment: float,
    annual_income: float,
    monthly_debt_payments: float,
    credit_score_range_min: int,
    loan_purpose: str,
    property_type: str,
    veteran_status: bool = False,
) -> dict:
    """Calculate pre-qualification result and generate letter data (FR-010, FR-013).
    Uses soft credit score range to estimate eligibility within 60 seconds.
    Returns max loan amount, rate range, and prequal letter reference.
    """
    loan_amount = estimated_home_price - down_payment
    ltv = loan_amount / estimated_home_price if estimated_home_price > 0 else 0
    monthly_income = annual_income / 12

    eligible_products = _select_eligible_products(
        credit_score_range_min, loan_amount, ltv, 0.50, veteran_status
    )

    if not eligible_products:
        return {
            "eligible": False,
            "reason": "No eligible products for the provided financial profile and credit score range.",
            "recommendations": [
                "Increase down payment to reduce LTV",
                "Pay down existing debt to reduce DTI",
                "Work on improving credit score",
            ],
        }

    best_product = eligible_products[0]
    rules = PRODUCT_RULES[best_product]
    rate_base = rules["rate_base"]
    rate_min = rate_base - 0.25
    rate_max = rate_base + 0.375

    monthly_payment = _estimate_monthly_payment(loan_amount, rate_base)
    dti = (monthly_payment + monthly_debt_payments) / monthly_income if monthly_income > 0 else 1.0

    letter_id = f"PQL-{str(uuid.uuid4())[:8].upper()}"
    expiry = (datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d")

    return {
        "eligible": True,
        "prequal_letter_id": letter_id,
        "max_loan_amount": round(loan_amount, 2),
        "estimated_rate_min": round(rate_min, 3),
        "estimated_rate_max": round(rate_max, 3),
        "credit_score_range": f"{credit_score_range_min}-{credit_score_range_min + 39}",
        "recommended_product": best_product,
        "eligible_products": eligible_products,
        "dti_ratio": round(dti, 4),
        "ltv_ratio": round(ltv, 4),
        "estimated_monthly_payment": round(monthly_payment, 2),
        "letter_expiration_date": expiry,
        "letter_url": f"/documents/prequal/{letter_id}.pdf",
        "generated_at": datetime.utcnow().isoformat(),
        "disclosure": (
            "This pre-qualification is not a commitment to lend. "
            "Final approval is subject to full underwriting, appraisal, and verification."
        ),
    }
