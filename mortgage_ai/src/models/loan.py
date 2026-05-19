"""Pydantic domain models for the B2C mortgage platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class LoanStage(str, Enum):
    """Maps to BRD Section 3 user journey stages."""
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


class LoanPurpose(str, Enum):
    PURCHASE = "purchase"
    REFINANCE = "refinance"


class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    MULTI_FAMILY_2_4 = "multi_family_2_4"
    MANUFACTURED = "manufactured"


class LoanProduct(str, Enum):
    CONVENTIONAL_30_FIXED = "conventional_30_fixed"
    CONVENTIONAL_15_FIXED = "conventional_15_fixed"
    CONVENTIONAL_ARM_5_1 = "conventional_arm_5_1"
    FHA_30_FIXED = "fha_30_fixed"
    VA_30_FIXED = "va_30_fixed"
    JUMBO_30_FIXED = "jumbo_30_fixed"


class DocumentType(str, Enum):
    PAY_STUB = "pay_stub"
    W2 = "w2"
    TAX_RETURN_1040 = "tax_return_1040"
    BANK_STATEMENT = "bank_statement"
    PURCHASE_CONTRACT = "purchase_contract"
    GOVERNMENT_ID = "government_id"
    INSURANCE_DECLARATION = "insurance_declaration"
    HOA_STATEMENT = "hoa_statement"
    OTHER = "other"


class AUSDecision(str, Enum):
    APPROVE_ELIGIBLE = "Approve/Eligible"
    REFER = "Refer"
    REFER_WITH_CAUTION = "Refer with Caution"
    OUT_OF_SCOPE = "Out of Scope"


# ── Domain Models ──────────────────────────────────────────────────────────────

class BorrowerInfo(BaseModel):
    borrower_id: str = Field(default_factory=lambda: str(uuid4()))
    full_name: str
    email: str
    phone: str
    ssn_last4: Optional[str] = None
    date_of_birth: Optional[str] = None
    annual_income: float = 0.0
    employment_status: str = "employed"
    employer_name: Optional[str] = None
    years_employed: Optional[float] = None
    monthly_debt_payments: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PrequalResult(BaseModel):
    eligible: bool
    max_loan_amount: float
    estimated_rate_min: float
    estimated_rate_max: float
    credit_score_range: str        # e.g., "720-759"
    dti_ratio: float
    ltv_ratio: float
    estimated_monthly_payment: float
    letter_url: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expiration_date: Optional[str] = None
    ineligibility_reasons: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_type: DocumentType
    filename: str
    file_size_kb: float
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    ocr_extracted: bool = False
    verified: bool = False
    rejection_reason: Optional[str] = None
    extracted_data: dict = Field(default_factory=dict)


class SignatureRecord(BaseModel):
    signature_id: str = Field(default_factory=lambda: str(uuid4()))
    document_name: str
    signed_at: datetime = Field(default_factory=datetime.utcnow)
    ip_address: str
    user_agent: Optional[str] = None
    tamper_evident_hash: str


class ClosingDetails(BaseModel):
    closing_date: Optional[str] = None
    closing_disclosure_sent_at: Optional[datetime] = None
    ron_session_url: Optional[str] = None
    title_company: Optional[str] = None
    settlement_agent: Optional[str] = None


class LoanApplication(BaseModel):
    """Full loan application record — persisted to DB."""
    application_id: str = Field(default_factory=lambda: str(uuid4()))
    borrower: BorrowerInfo
    co_borrower: Optional[BorrowerInfo] = None

    # Loan request
    loan_purpose: LoanPurpose = LoanPurpose.PURCHASE
    property_type: PropertyType = PropertyType.SINGLE_FAMILY
    estimated_home_price: float = 0.0
    down_payment: float = 0.0
    loan_amount: float = 0.0

    # Journey tracking
    stage: LoanStage = LoanStage.DISCOVERY
    urla_sections_completed: list[str] = Field(default_factory=list)
    hmda_data_collected: bool = False

    # Pre-qualification
    prequal_result: Optional[PrequalResult] = None
    hard_pull_consented: bool = False
    credit_score: Optional[int] = None

    # Property & product
    property_address: Optional[str] = None
    appraisal_value: Optional[float] = None
    selected_product: Optional[LoanProduct] = None
    selected_term_years: Optional[int] = None
    interest_rate: Optional[float] = None
    rate_locked: bool = False
    monthly_payment: Optional[float] = None
    closing_costs: Optional[float] = None
    apr: Optional[float] = None

    # Documents
    documents: list[DocumentRecord] = Field(default_factory=list)
    documents_verified: bool = False
    missing_documents: list[str] = Field(default_factory=list)

    # Compliance
    e_consent_given: bool = False
    disclosures_delivered: list[str] = Field(default_factory=list)
    signatures: list[SignatureRecord] = Field(default_factory=list)

    # Underwriting
    aus_decision: Optional[AUSDecision] = None
    conditions: list[str] = Field(default_factory=list)
    conditions_cleared: list[str] = Field(default_factory=list)

    # Closing
    closing: Optional[ClosingDetails] = None

    # Compliance flags & system
    compliance_flags: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    error: Optional[str] = None

    # KPI timestamps (BRD §1.1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    prequal_completed_at: Optional[datetime] = None
    conditional_approval_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    @property
    def dti_ratio(self) -> float:
        monthly_income = self.borrower.annual_income / 12
        if monthly_income == 0:
            return 0.0
        monthly_payment = self.monthly_payment or 0.0
        monthly_debt = self.borrower.monthly_debt_payments
        return (monthly_payment + monthly_debt) / monthly_income

    @property
    def ltv_ratio(self) -> float:
        home_value = self.appraisal_value or self.estimated_home_price
        if home_value == 0:
            return 0.0
        return self.loan_amount / home_value

    @property
    def hours_to_conditional(self) -> Optional[float]:
        if self.conditional_approval_at and self.created_at:
            delta = self.conditional_approval_at - self.created_at
            return delta.total_seconds() / 3600
        return None
