"""Credit bureau tools — FR-011, FR-060.

Sandbox stubs simulate Equifax / Experian / TransUnion responses.
In production, replace _call_bureau() with real API calls.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SoftCreditInput(BaseModel):
    borrower_name: str = Field(description="Full legal name of the borrower")
    ssn_last4: str = Field(description="Last 4 digits of SSN for identity matching")
    annual_income: float = Field(description="Self-reported annual gross income in USD")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")


class HardCreditInput(BaseModel):
    borrower_name: str
    ssn_last4: str
    annual_income: float
    date_of_birth: str
    application_id: str = Field(description="Mortgage application ID for audit trail")
    consented: bool = Field(description="Borrower explicitly consented to hard pull (FCRA)")


def _simulate_credit_profile(annual_income: float) -> dict:
    """Deterministically simulate a credit profile based on income tier."""
    if annual_income >= 150_000:
        score = random.randint(740, 850)
        score_range = "740-850"
        derogatory_marks = 0
    elif annual_income >= 80_000:
        score = random.randint(680, 739)
        score_range = "680-739"
        derogatory_marks = random.randint(0, 1)
    elif annual_income >= 50_000:
        score = random.randint(620, 679)
        score_range = "620-679"
        derogatory_marks = random.randint(0, 2)
    else:
        score = random.randint(580, 619)
        score_range = "580-619"
        derogatory_marks = random.randint(1, 3)

    return {
        "score": score,
        "score_range": score_range,
        "open_accounts": random.randint(3, 12),
        "total_balance": random.randint(5000, 80000),
        "derogatory_marks": derogatory_marks,
        "inquiries_12mo": random.randint(0, 4),
        "oldest_account_years": random.randint(2, 20),
        "credit_utilization_pct": random.randint(5, 45),
    }


@tool("soft_credit_check", args_schema=SoftCreditInput, return_direct=False)
def soft_credit_check(
    borrower_name: str,
    ssn_last4: str,
    annual_income: float,
    date_of_birth: str,
) -> dict:
    """Perform a SOFT credit inquiry (no impact to credit score) to retrieve
    the borrower's credit score range for pre-qualification purposes (FR-011).
    Returns score range, estimated eligibility, and inquiry reference ID.
    Does NOT affect the borrower's credit score.
    """
    profile = _simulate_credit_profile(annual_income)
    inquiry_id = str(uuid.uuid4())

    return {
        "inquiry_type": "soft",
        "inquiry_id": inquiry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "score_range": profile["score_range"],
        "estimated_score": profile["score"],
        "open_accounts": profile["open_accounts"],
        "derogatory_marks": profile["derogatory_marks"],
        "credit_utilization_pct": profile["credit_utilization_pct"],
        "inquiries_12mo": profile["inquiries_12mo"],
        "score_impact": "none — soft inquiry only",
        "eligible_for_prequal": profile["score"] >= 580,
    }


@tool("hard_credit_check", args_schema=HardCreditInput, return_direct=False)
def hard_credit_check(
    borrower_name: str,
    ssn_last4: str,
    annual_income: float,
    date_of_birth: str,
    application_id: str,
    consented: bool,
) -> dict:
    """Perform a HARD credit inquiry (impacts credit score by ~5 pts) for
    full underwriting. Requires explicit borrower consent (FCRA, FR-011).
    Returns full credit report reference for AUS submission.
    """
    if not consented:
        return {
            "error": "FCRA_CONSENT_REQUIRED",
            "message": "Hard credit pull requires explicit borrower consent. Obtain consent before proceeding.",
        }

    profile = _simulate_credit_profile(annual_income)
    report_id = f"CR-{str(uuid.uuid4())[:8].upper()}"

    return {
        "inquiry_type": "hard",
        "inquiry_id": str(uuid.uuid4()),
        "credit_report_id": report_id,
        "application_id": application_id,
        "timestamp": datetime.utcnow().isoformat(),
        "fico_score": profile["score"],
        "score_range": profile["score_range"],
        "open_accounts": profile["open_accounts"],
        "total_balance": profile["total_balance"],
        "derogatory_marks": profile["derogatory_marks"],
        "credit_utilization_pct": profile["credit_utilization_pct"],
        "inquiries_12mo": profile["inquiries_12mo"],
        "oldest_account_years": profile["oldest_account_years"],
        "monthly_obligations_estimated": profile["total_balance"] * 0.02,
        "score_impact": "~5 points (hard inquiry)",
        "report_valid_days": 120,
        "bureaus_queried": ["Equifax", "Experian", "TransUnion"],
    }
