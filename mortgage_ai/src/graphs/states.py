"""LangGraph state definitions for the B2C mortgage workflow.

MortgageState is the single source of truth flowing through the graph.
It is persisted by the MemorySaver (or SqliteSaver in production) so borrowers
can pause and resume across devices (FR-014, FR-022).
"""

from __future__ import annotations

from typing import Annotated, Optional
from enum import Enum

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class LoanStage(str, Enum):
    """Nine journey stages from BRD §3 plus terminal states."""
    DISCOVERY = "discovery"
    PREQUALIFICATION = "prequalification"
    APPLICATION = "application"
    DOCUMENT_UPLOAD = "document_upload"
    PROPERTY_LOAN_SELECTION = "property_loan_selection"
    DISCLOSURES = "disclosures"
    UNDERWRITING = "underwriting"
    CLOSING = "closing"
    POST_SUBMISSION = "post_submission"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    DECLINED = "declined"


# Stage progression order (happy path)
STAGE_ORDER = [
    LoanStage.DISCOVERY,
    LoanStage.PREQUALIFICATION,
    LoanStage.APPLICATION,
    LoanStage.DOCUMENT_UPLOAD,
    LoanStage.PROPERTY_LOAN_SELECTION,
    LoanStage.DISCLOSURES,
    LoanStage.UNDERWRITING,
    LoanStage.CLOSING,
    LoanStage.POST_SUBMISSION,
    LoanStage.COMPLETED,
]


class MortgageState(dict):
    """TypedDict-compatible state for LangGraph.

    Uses Annotated[list, add_messages] for the messages field so that
    LangGraph's message-append reducer works correctly.

    All fields have defaults so partial updates are safe.
    """
    # -- Core identifiers --
    application_id: str
    borrower_id: str
    stage: str                          # LoanStage value

    # -- Conversation history (append-only via add_messages reducer) --
    messages: Annotated[list[AnyMessage], add_messages]

    # -- Borrower profile --
    borrower_name: str
    borrower_email: str
    borrower_phone: str
    annual_income: float
    monthly_debt_payments: float
    employment_status: str
    co_borrower_present: bool
    co_borrower_name: Optional[str]
    veteran_status: bool

    # -- Loan request --
    loan_purpose: str                   # purchase | refinance
    property_type: str
    estimated_home_price: float
    down_payment: float
    loan_amount: float

    # -- Pre-qualification --
    credit_score_range_min: int
    credit_score_range_str: str         # e.g., "720-759"
    prequal_eligible: Optional[bool]
    prequal_max_loan: float
    prequal_rate_min: float
    prequal_rate_max: float
    prequal_letter_url: Optional[str]
    prequal_dti: float
    prequal_ltv: float
    prequal_eligible_products: list[str]

    # -- URLA / Form 1003 --
    urla_sections_completed: list[str]
    urla_complete: bool
    hmda_opted_out: bool
    hmda_complete: bool

    # -- Credit --
    hard_pull_consented: bool
    credit_score: Optional[int]
    credit_report_id: Optional[str]
    aus_ready: bool

    # -- Documents --
    documents_uploaded: list[dict]      # {document_id, doc_type, filename, verified}
    documents_verified: bool
    missing_documents: list[str]

    # -- Property & loan selection --
    property_address: Optional[str]
    appraisal_value: Optional[float]
    selected_product: Optional[str]
    selected_term_years: Optional[int]
    interest_rate: Optional[float]
    rate_locked: bool
    rate_lock_id: Optional[str]
    monthly_payment: Optional[float]
    closing_costs: Optional[float]
    apr: Optional[float]

    # -- Disclosures & e-signature --
    e_consent_given: bool
    e_consent_id: Optional[str]
    disclosures_delivered: list[str]    # names of delivered disclosures
    signatures_collected: list[str]     # disclosure names that have been signed

    # -- Underwriting --
    aus_decision: Optional[str]         # AUSDecision enum value
    conditions: list[str]
    conditions_cleared: list[str]

    # -- Closing --
    closing_date: Optional[str]
    closing_disclosure_sent: bool
    ron_available: bool

    # -- System / orchestration --
    next_action: str                    # hint for the graph router
    error: Optional[str]
    human_review_required: bool
    compliance_flags: list[str]
    loan_officer_id: Optional[str]

    # -- KPI timestamps (ISO strings) --
    created_at: str
    prequal_completed_at: Optional[str]
    conditional_approval_at: Optional[str]
    closed_at: Optional[str]


def default_state(application_id: str, borrower_id: str, borrower_name: str, borrower_email: str) -> dict:
    """Return a fully-initialized MortgageState dict with safe defaults."""
    from datetime import datetime
    return {
        "application_id": application_id,
        "borrower_id": borrower_id,
        "stage": LoanStage.DISCOVERY.value,
        "messages": [],
        "borrower_name": borrower_name,
        "borrower_email": borrower_email,
        "borrower_phone": "",
        "annual_income": 0.0,
        "monthly_debt_payments": 0.0,
        "employment_status": "employed",
        "co_borrower_present": False,
        "co_borrower_name": None,
        "veteran_status": False,
        "loan_purpose": "purchase",
        "property_type": "single_family",
        "estimated_home_price": 0.0,
        "down_payment": 0.0,
        "loan_amount": 0.0,
        "credit_score_range_min": 0,
        "credit_score_range_str": "",
        "prequal_eligible": None,
        "prequal_max_loan": 0.0,
        "prequal_rate_min": 0.0,
        "prequal_rate_max": 0.0,
        "prequal_letter_url": None,
        "prequal_dti": 0.0,
        "prequal_ltv": 0.0,
        "prequal_eligible_products": [],
        "urla_sections_completed": [],
        "urla_complete": False,
        "hmda_opted_out": False,
        "hmda_complete": False,
        "hard_pull_consented": False,
        "credit_score": None,
        "credit_report_id": None,
        "aus_ready": False,
        "documents_uploaded": [],
        "documents_verified": False,
        "missing_documents": [],
        "property_address": None,
        "appraisal_value": None,
        "selected_product": None,
        "selected_term_years": None,
        "interest_rate": None,
        "rate_locked": False,
        "rate_lock_id": None,
        "monthly_payment": None,
        "closing_costs": None,
        "apr": None,
        "e_consent_given": False,
        "e_consent_id": None,
        "disclosures_delivered": [],
        "signatures_collected": [],
        "aus_decision": None,
        "conditions": [],
        "conditions_cleared": [],
        "closing_date": None,
        "closing_disclosure_sent": False,
        "ron_available": False,
        "next_action": "start",
        "error": None,
        "human_review_required": False,
        "compliance_flags": [],
        "loan_officer_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "prequal_completed_at": None,
        "conditional_approval_at": None,
        "closed_at": None,
    }
