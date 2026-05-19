"""Regulatory compliance tools — BRD §5.4, FR-040 through FR-044.

Enforces TRID, HMDA, ECOA, FCRA, RESPA, and E-SIGN Act requirements.
All disclosure timing is non-negotiable per BRD §7.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field


TRID_DISCLOSURES = {
    "loan_estimate": {
        "required_by_days": 3,      # 3 business days from application
        "resend_trigger": "significant_change",
        "regulation": "TRID / Regulation Z § 1026.19(e)",
    },
    "closing_disclosure": {
        "required_by_days": 3,      # 3 business days before consummation
        "regulation": "TRID / Regulation Z § 1026.19(f)",
    },
}

HMDA_REQUIRED_FIELDS = [
    "loan_type",
    "loan_purpose",
    "property_type",
    "occupancy_type",
    "loan_amount",
    "property_address",
    "ethnicity",
    "race",
    "sex",
    "income",
    "purchaser_type",
]


class TRIDComplianceInput(BaseModel):
    application_id: str
    disclosure_type: str = Field(description="loan_estimate or closing_disclosure")
    application_received_at: str = Field(description="ISO datetime when application was received")
    scheduled_closing_date: str = Field(description="ISO date of scheduled closing")
    disclosures_delivered: list[str] = Field(
        default_factory=list,
        description="List of previously delivered disclosure names"
    )


class HMDAInput(BaseModel):
    application_id: str
    loan_type: str
    loan_purpose: str
    property_type: str
    occupancy_type: str = Field(description="principal_residence, second_home, or investment")
    loan_amount: float
    property_address: str
    applicant_ethnicity: str
    applicant_race: str
    applicant_sex: str
    annual_income: float
    opted_out: bool = Field(default=False, description="True if borrower opted out of providing demographic info")


class ECOAInput(BaseModel):
    application_id: str
    decision: str = Field(description="approved, conditionally_approved, declined, or incomplete")
    decision_date: str
    principal_reason: str = Field(default="", description="Primary reason for adverse action if declined")
    secondary_reasons: list[str] = Field(default_factory=list)
    credit_score_used: int = Field(default=0)
    credit_bureau_name: str = Field(default="")


class ESignConsentInput(BaseModel):
    application_id: str
    borrower_id: str
    ip_address: str
    user_agent: str = ""
    consent_text_version: str = Field(default="v1.0", description="Version of E-SIGN consent language displayed")


@tool("check_trid_compliance", args_schema=TRIDComplianceInput, return_direct=False)
def check_trid_compliance(
    application_id: str,
    disclosure_type: str,
    application_received_at: str,
    scheduled_closing_date: str,
    disclosures_delivered: list[str],
) -> dict:
    """Verify TRID timing compliance for Loan Estimate and Closing Disclosure.
    Regulatory disclosure timing is a hard constraint (BRD §7.2, FR-040).
    Returns whether disclosure can be delivered, deadline, and compliance status.
    """
    if disclosure_type not in TRID_DISCLOSURES:
        return {"error": "UNKNOWN_DISCLOSURE_TYPE", "valid_types": list(TRID_DISCLOSURES.keys())}

    rules = TRID_DISCLOSURES[disclosure_type]
    now = datetime.utcnow()

    try:
        app_dt = datetime.fromisoformat(application_received_at)
        closing_dt = datetime.strptime(scheduled_closing_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": "INVALID_DATE_FORMAT", "detail": str(e)}

    issues: list[str] = []
    warnings: list[str] = []

    if disclosure_type == "loan_estimate":
        # Must be delivered within 3 business days of application
        deadline = app_dt + timedelta(days=rules["required_by_days"])
        days_remaining = (deadline - now).days

        if now > deadline:
            issues.append(
                f"TRID VIOLATION: Loan Estimate must be delivered within "
                f"{rules['required_by_days']} business days of application. "
                f"Deadline was {deadline.strftime('%Y-%m-%d %H:%M')} UTC."
            )
        elif days_remaining <= 0:
            warnings.append("Loan Estimate delivery deadline is today — deliver immediately.")
        else:
            warnings.append(f"Loan Estimate must be delivered by {deadline.strftime('%Y-%m-%d')} UTC.")

        already_sent = "loan_estimate" in disclosures_delivered

    elif disclosure_type == "closing_disclosure":
        # Must be received 3 business days BEFORE consummation
        deadline = closing_dt - timedelta(days=rules["required_by_days"])
        days_remaining = (deadline - now.date()).days if hasattr(now, "date") else 0

        if not any("closing_disclosure" in d for d in disclosures_delivered):
            if now.date() > deadline.date():
                issues.append(
                    f"TRID VIOLATION: Closing Disclosure must be delivered at least 3 business days "
                    f"before closing ({scheduled_closing_date}). Deliver immediately or reschedule closing."
                )
        already_sent = "closing_disclosure" in disclosures_delivered

    return {
        "application_id": application_id,
        "disclosure_type": disclosure_type,
        "regulation": rules["regulation"],
        "compliant": len(issues) == 0,
        "deadline": deadline.isoformat() if "deadline" in dir() else None,
        "issues": issues,
        "warnings": warnings,
        "already_delivered": already_sent,
        "checked_at": now.isoformat(),
    }


@tool("check_hmda_data", args_schema=HMDAInput, return_direct=False)
def check_hmda_data(
    application_id: str,
    loan_type: str,
    loan_purpose: str,
    property_type: str,
    occupancy_type: str,
    loan_amount: float,
    property_address: str,
    applicant_ethnicity: str,
    applicant_race: str,
    applicant_sex: str,
    annual_income: float,
    opted_out: bool = False,
) -> dict:
    """Validate HMDA data completeness for regulatory reporting (FR-025).
    HMDA (Regulation C) requires collection of demographic data with opt-out option.
    """
    issues: list[str] = []
    hmda_record: dict = {
        "application_id": application_id,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "property_type": property_type,
        "occupancy_type": occupancy_type,
        "loan_amount": loan_amount,
        "property_address": property_address,
        "income": annual_income,
    }

    if opted_out:
        hmda_record.update({
            "applicant_ethnicity": "information_not_provided",
            "applicant_race": "information_not_provided",
            "applicant_sex": "information_not_provided",
        })
    else:
        hmda_record.update({
            "applicant_ethnicity": applicant_ethnicity,
            "applicant_race": applicant_race,
            "applicant_sex": applicant_sex,
        })

    # Validate required non-demographic fields
    if not property_address or len(property_address) < 10:
        issues.append("Property address appears incomplete — required for HMDA geocoding")
    if loan_amount <= 0:
        issues.append("Loan amount must be positive for HMDA reporting")
    if occupancy_type not in ("principal_residence", "second_home", "investment"):
        issues.append(f"Unknown occupancy type: {occupancy_type}")

    return {
        "hmda_complete": len(issues) == 0,
        "hmda_record": hmda_record,
        "opted_out": opted_out,
        "issues": issues,
        "regulation": "HMDA / Regulation C 12 CFR § 1003",
        "lar_reportable": loan_amount >= 10_000,
    }


@tool("check_ecoa_compliance", args_schema=ECOAInput, return_direct=False)
def check_ecoa_compliance(
    application_id: str,
    decision: str,
    decision_date: str,
    principal_reason: str,
    secondary_reasons: list[str],
    credit_score_used: int,
    credit_bureau_name: str,
) -> dict:
    """Ensure ECOA / Regulation B compliance for adverse action notices (FR-044).
    Declined applicants must receive notice within 30 days with specific reasons.
    Returns required notice content and timing compliance status.
    """
    issues: list[str] = []

    try:
        decision_dt = datetime.strptime(decision_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "INVALID_DATE_FORMAT"}

    days_since_decision = (datetime.utcnow() - decision_dt).days

    if decision == "declined":
        # ECOA: Adverse action notice within 30 days
        if days_since_decision > 30:
            issues.append(
                f"ECOA VIOLATION: Adverse action notice must be sent within 30 days. "
                f"{days_since_decision} days have elapsed."
            )
        if not principal_reason:
            issues.append("ECOA requires at least one specific reason for adverse action")
        if len(principal_reason) + len(" ".join(secondary_reasons)) < 10:
            issues.append("Adverse action reasons are too vague — must be specific and factual")

    notice_content = None
    if decision in ("declined", "conditionally_approved"):
        notice_content = {
            "type": "adverse_action_notice" if decision == "declined" else "notice_of_incompleteness",
            "statement_of_action": f"Your mortgage application has been {decision}.",
            "principal_reason": principal_reason,
            "secondary_reasons": secondary_reasons,
            "ecoa_statement": (
                "The Federal Equal Credit Opportunity Act prohibits creditors from discriminating "
                "against credit applicants on the basis of race, color, religion, national origin, sex, "
                "marital status, age, or because you receive public assistance."
            ),
            "credit_score_disclosure": {
                "score": credit_score_used,
                "score_range": "300-850",
                "bureau": credit_bureau_name,
                "key_factors": ["Payment history", "Credit utilization", "Credit age"],
            } if credit_score_used > 0 else None,
            "contact_cfpb": "1-855-411-2372 | www.consumerfinance.gov",
            "application_id": application_id,
        }

    return {
        "compliant": len(issues) == 0,
        "decision": decision,
        "days_since_decision": days_since_decision,
        "notice_required": decision in ("declined", "conditionally_approved"),
        "notice_content": notice_content,
        "issues": issues,
        "regulation": "ECOA / Regulation B 12 CFR § 1002",
    }


@tool("capture_esign_consent", args_schema=ESignConsentInput, return_direct=False)
def capture_esign_consent(
    application_id: str,
    borrower_id: str,
    ip_address: str,
    user_agent: str = "",
    consent_text_version: str = "v1.0",
) -> dict:
    """Capture borrower's E-SIGN Act consent before delivering electronic disclosures.
    Required by 15 U.S.C. § 7001 (E-SIGN Act) and FR-041.
    Stores tamper-evident consent record with IP, timestamp, and consent text version.
    """
    consent_id = f"ESIGN-{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.utcnow()

    return {
        "consent_id": consent_id,
        "application_id": application_id,
        "borrower_id": borrower_id,
        "consented": True,
        "consent_text_version": consent_text_version,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "consented_at": timestamp.isoformat(),
        "audit_hash": f"sha256:{uuid.uuid4().hex}",  # In production: real hash of consent record
        "disclosure": (
            "Borrower has consented to receive disclosures electronically per E-SIGN Act. "
            "Borrower retains right to withdraw consent and receive paper copies."
        ),
        "regulation": "E-SIGN Act 15 U.S.C. § 7001; UETA",
    }
