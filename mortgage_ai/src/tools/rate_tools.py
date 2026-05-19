"""Rate calculation and loan product selection tools — BRD §3.5, FR-010."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .eligibility_tools import PRODUCT_RULES, _estimate_monthly_payment


class PaymentCalcInput(BaseModel):
    loan_amount: float = Field(description="Principal loan amount in USD")
    annual_rate_pct: float = Field(description="Annual interest rate as percentage (e.g., 6.875)")
    term_years: int = Field(default=30, description="Loan term in years (15 or 30)")
    loan_product: str = Field(description="Loan product type for PMI/MIP calculation")
    down_payment: float = Field(default=0.0, description="Down payment amount for LTV-based PMI calc")
    home_value: float = Field(default=0.0, description="Home value for LTV calculation")


class APRCalcInput(BaseModel):
    loan_amount: float
    annual_rate_pct: float
    term_years: int = 30
    origination_fee_pct: float = Field(default=0.01, description="Origination fee as decimal (e.g., 0.01 for 1%)")
    discount_points: float = Field(default=0.0, description="Discount points paid upfront")
    other_fees: float = Field(default=1_500.0, description="Other financed fees (title, appraisal, etc.)")


class RateLockInput(BaseModel):
    application_id: str
    loan_amount: float
    loan_product: str
    rate_pct: float
    lock_period_days: int = Field(default=30, description="Rate lock period: 15, 30, 45, or 60 days")
    borrower_name: str


class CurrentRatesInput(BaseModel):
    loan_purpose: str = Field(description="purchase or refinance")
    credit_score: int = Field(description="Borrower FICO score for pricing")
    loan_amount: float
    home_value: float
    veteran_status: bool = False


def _closing_costs_estimate(loan_amount: float, home_value: float) -> dict:
    """Estimate closing costs breakdown (TRID disclosure prep)."""
    return {
        "origination_fee": round(loan_amount * 0.01, 2),
        "appraisal_fee": 600.0,
        "title_insurance": round(home_value * 0.005, 2),
        "title_search": 250.0,
        "recording_fees": 125.0,
        "prepaid_interest": round(loan_amount * 0.00019 * 15, 2),  # ~15 days
        "homeowners_insurance_prepaid": 1_200.0,
        "property_tax_escrow": round(home_value * 0.012 / 12 * 3, 2),  # 3-month escrow
    }


@tool("calculate_payment", args_schema=PaymentCalcInput, return_direct=False)
def calculate_payment(
    loan_amount: float,
    annual_rate_pct: float,
    term_years: int,
    loan_product: str,
    down_payment: float = 0.0,
    home_value: float = 0.0,
) -> dict:
    """Compute monthly P&I payment, PMI/MIP if applicable, taxes, insurance
    estimate (PITI), and total closing costs. Required for TRID Loan Estimate (FR-040).
    """
    pi_payment = _estimate_monthly_payment(loan_amount, annual_rate_pct, term_years)

    # PMI / MIP
    ltv = loan_amount / home_value if home_value > 0 else 1.0
    pmi_monthly = 0.0
    mip_monthly = 0.0

    rules = PRODUCT_RULES.get(loan_product, {})
    if "pmi_required_ltv" in rules and ltv > rules["pmi_required_ltv"]:
        pmi_monthly = round(loan_amount * 0.0085 / 12, 2)

    if "mip_annual" in rules:
        mip_monthly = round(loan_amount * rules["mip_annual"] / 12, 2)

    # Estimated escrow (taxes + insurance)
    monthly_taxes = round((home_value * 0.012) / 12, 2) if home_value > 0 else 300.0
    monthly_insurance = round((home_value * 0.006) / 12, 2) if home_value > 0 else 100.0

    total_piti = round(pi_payment + pmi_monthly + mip_monthly + monthly_taxes + monthly_insurance, 2)
    closing_costs = _closing_costs_estimate(loan_amount, home_value)
    total_closing = round(sum(closing_costs.values()), 2)

    return {
        "loan_amount": loan_amount,
        "annual_rate_pct": annual_rate_pct,
        "term_years": term_years,
        "monthly_pi": round(pi_payment, 2),
        "monthly_pmi": pmi_monthly,
        "monthly_mip": mip_monthly,
        "monthly_taxes_estimated": monthly_taxes,
        "monthly_insurance_estimated": monthly_insurance,
        "total_monthly_piti": total_piti,
        "closing_costs_breakdown": closing_costs,
        "total_closing_costs": total_closing,
        "ltv_ratio": round(ltv, 4),
        "pmi_removal_at_ltv": 0.80 if pmi_monthly > 0 else None,
    }


@tool("calculate_apr", args_schema=APRCalcInput, return_direct=False)
def calculate_apr(
    loan_amount: float,
    annual_rate_pct: float,
    term_years: int,
    origination_fee_pct: float,
    discount_points: float,
    other_fees: float,
) -> dict:
    """Calculate Annual Percentage Rate (APR) including all financed costs.
    Required for TILA / Regulation Z disclosure (TRID, FR-040).
    """
    monthly_rate = annual_rate_pct / 100 / 12
    n = term_years * 12
    monthly_payment = _estimate_monthly_payment(loan_amount, annual_rate_pct, term_years)

    total_fees = (loan_amount * origination_fee_pct) + (loan_amount * discount_points * 0.01) + other_fees
    adjusted_principal = loan_amount - total_fees

    # Newton-Raphson APR solve
    apr_monthly = monthly_rate
    for _ in range(100):
        pv = monthly_payment * (1 - (1 + apr_monthly) ** -n) / apr_monthly
        dpv = (
            monthly_payment * n * (1 + apr_monthly) ** (-n - 1) / apr_monthly
            - monthly_payment * (1 - (1 + apr_monthly) ** -n) / apr_monthly**2
        )
        delta = (pv - adjusted_principal) / dpv
        apr_monthly -= delta
        if abs(delta) < 1e-10:
            break

    apr_annual = apr_monthly * 12 * 100

    return {
        "note_rate_pct": round(annual_rate_pct, 3),
        "apr_pct": round(apr_annual, 3),
        "apr_spread_bps": round((apr_annual - annual_rate_pct) * 100, 1),
        "total_financed_fees": round(total_fees, 2),
        "total_interest_paid": round(monthly_payment * n - loan_amount, 2),
        "total_cost_of_loan": round(monthly_payment * n + total_fees, 2),
        "disclosure_note": "APR disclosed per Regulation Z. Higher than note rate due to included fees.",
    }


@tool("get_current_rates", args_schema=CurrentRatesInput, return_direct=False)
def get_current_rates(
    loan_purpose: str,
    credit_score: int,
    loan_amount: float,
    home_value: float,
    veteran_status: bool = False,
) -> dict:
    """Retrieve current mortgage rate options for the borrower's profile.
    Rates are risk-adjusted based on FICO, LTV, loan purpose, and product type.
    """
    ltv = loan_amount / home_value if home_value > 0 else 0.80
    rate_adjustments = 0.0

    # FICO-based adjustments (LLPA proxies)
    if credit_score < 640:
        rate_adjustments += 0.75
    elif credit_score < 680:
        rate_adjustments += 0.50
    elif credit_score < 720:
        rate_adjustments += 0.25
    elif credit_score >= 760:
        rate_adjustments -= 0.125

    # LTV adjustments
    if ltv > 0.95:
        rate_adjustments += 0.375
    elif ltv > 0.90:
        rate_adjustments += 0.25
    elif ltv > 0.80:
        rate_adjustments += 0.125

    # Refi premium
    if loan_purpose == "refinance":
        rate_adjustments += 0.125

    products = []
    for product, rules in PRODUCT_RULES.items():
        if product == "va_30_fixed" and not veteran_status:
            continue
        if credit_score < rules["min_fico"]:
            continue
        adj_rate = rules["rate_base"] + rate_adjustments
        products.append({
            "product": product,
            "interest_rate": round(adj_rate, 3),
            "apr": round(adj_rate + 0.15, 3),
            "points": 0.0,
            "monthly_payment_estimate": round(
                _estimate_monthly_payment(loan_amount, adj_rate), 2
            ),
        })

    return {
        "effective_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "rate_lock_available": True,
        "available_products": products,
        "rate_disclaimer": (
            "Rates shown are illustrative and subject to change. "
            "Lock rate to secure pricing. Final rate may differ based on full underwriting."
        ),
    }


@tool("lock_rate", args_schema=RateLockInput, return_direct=False)
def lock_rate(
    application_id: str,
    loan_amount: float,
    loan_product: str,
    rate_pct: float,
    lock_period_days: int,
    borrower_name: str,
) -> dict:
    """Lock the interest rate for the borrower's application.
    Prevents rate fluctuation during underwriting (BRD §3.5, FR-064).
    Returns lock confirmation with expiration timestamp.
    """
    if lock_period_days not in (15, 30, 45, 60):
        return {"error": "INVALID_LOCK_PERIOD", "message": "Lock period must be 15, 30, 45, or 60 days"}

    # Longer locks cost more (pricing spread)
    lock_premiums = {15: 0.0, 30: 0.0, 45: 0.125, 60: 0.25}
    locked_rate = round(rate_pct + lock_premiums[lock_period_days] / 100, 4)

    lock_id = f"LCK-{str(uuid.uuid4())[:8].upper()}"
    expiration = datetime.utcnow() + timedelta(days=lock_period_days)

    return {
        "lock_id": lock_id,
        "application_id": application_id,
        "borrower_name": borrower_name,
        "locked_rate_pct": locked_rate,
        "lock_period_days": lock_period_days,
        "locked_at": datetime.utcnow().isoformat(),
        "expires_at": expiration.isoformat(),
        "loan_product": loan_product,
        "loan_amount": loan_amount,
        "float_down_option": lock_period_days >= 45,
        "confirmation_message": (
            f"Rate locked at {locked_rate:.3f}% for {lock_period_days} days. "
            f"Lock expires {expiration.strftime('%B %d, %Y')}."
        ),
    }
