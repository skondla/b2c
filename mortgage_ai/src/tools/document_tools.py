"""Document classification and verification tools — FR-030 through FR-034.

Simulates OCR / ML document classification pipeline.
In production: replace with AWS Textract, Google Document AI, or Azure Form Recognizer.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field


DOCUMENT_PATTERNS = {
    "pay_stub": {
        "keywords": ["pay period", "gross pay", "net pay", "employer", "ytd"],
        "required_fields": ["employer_name", "pay_period_end", "gross_pay", "net_pay", "ytd_gross"],
        "validity_days": 60,
    },
    "w2": {
        "keywords": ["wages tips", "federal income tax", "box 1", "employer ein"],
        "required_fields": ["employer_name", "employee_name", "tax_year", "box1_wages", "employer_ein"],
        "validity_days": 365,
    },
    "tax_return_1040": {
        "keywords": ["1040", "adjusted gross income", "taxable income", "irs"],
        "required_fields": ["tax_year", "agi", "total_income", "filing_status"],
        "validity_days": 730,
    },
    "bank_statement": {
        "keywords": ["account number", "routing", "beginning balance", "ending balance"],
        "required_fields": ["bank_name", "account_last4", "statement_date", "ending_balance"],
        "validity_days": 60,
    },
    "purchase_contract": {
        "keywords": ["purchase price", "seller", "buyer", "closing date", "earnest money"],
        "required_fields": ["property_address", "purchase_price", "closing_date", "seller_name", "buyer_name"],
        "validity_days": 365,
    },
    "government_id": {
        "keywords": ["date of birth", "expiration", "license", "passport", "id number"],
        "required_fields": ["full_name", "date_of_birth", "expiration_date", "id_number"],
        "validity_days": None,  # checked against expiration date
    },
}


class ClassifyDocumentInput(BaseModel):
    filename: str = Field(description="Original filename of the uploaded document")
    file_size_kb: float = Field(description="File size in kilobytes")
    raw_text_excerpt: str = Field(description="First 500 characters extracted from document via OCR")


class ExtractDocumentInput(BaseModel):
    document_id: str = Field(description="Internal document ID assigned at upload")
    doc_type: str = Field(description="Document type as classified by classify_document tool")
    raw_text: str = Field(description="Full OCR-extracted text from the document")


class ValidateDocumentInput(BaseModel):
    document_id: str
    doc_type: str
    extracted_data: dict = Field(description="Key-value pairs extracted from document")
    borrower_name: str = Field(description="Expected borrower name for matching")
    annual_income_declared: float = Field(description="Income declared in application for consistency check")


def _classify_from_text(text: str) -> tuple[str, float]:
    """Score each doc type by keyword frequency."""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for doc_type, meta in DOCUMENT_PATTERNS.items():
        hits = sum(1 for kw in meta["keywords"] if kw in text_lower)
        scores[doc_type] = hits / len(meta["keywords"])
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


@tool("classify_document", args_schema=ClassifyDocumentInput, return_direct=False)
def classify_document(filename: str, file_size_kb: float, raw_text_excerpt: str) -> dict:
    """Classify an uploaded document using OCR text analysis (FR-031).
    Identifies document type (pay stub, W-2, bank statement, etc.) and
    returns classification confidence. Flags illegible or unknown documents.
    """
    if file_size_kb > 25 * 1024:
        return {
            "error": "FILE_TOO_LARGE",
            "message": f"File size {file_size_kb:.0f} KB exceeds 25 MB limit.",
        }

    if len(raw_text_excerpt.strip()) < 20:
        return {
            "classified": False,
            "rejection_reason": "ILLEGIBLE",
            "message": "Document appears blank or illegible. Please re-scan or photograph in better lighting.",
            "action_required": "resubmit",
        }

    doc_type, confidence = _classify_from_text(raw_text_excerpt)

    # Low confidence → flag for review
    if confidence < 0.20:
        return {
            "classified": False,
            "rejection_reason": "UNKNOWN_DOCUMENT_TYPE",
            "suggested_type": doc_type,
            "confidence": round(confidence, 3),
            "message": "Could not reliably classify this document. Please verify the document type.",
            "action_required": "confirm_type",
        }

    return {
        "classified": True,
        "doc_type": doc_type,
        "confidence": round(confidence, 3),
        "filename": filename,
        "file_size_kb": file_size_kb,
        "requires_human_review": confidence < 0.50,
        "message": f"Document classified as '{doc_type}' with {confidence:.0%} confidence.",
    }


@tool("extract_document_data", args_schema=ExtractDocumentInput, return_direct=False)
def extract_document_data(document_id: str, doc_type: str, raw_text: str) -> dict:
    """Extract structured key-value data from a classified document via OCR (FR-031).
    Returns financial figures, dates, and entity names needed for application verification.
    """
    if doc_type not in DOCUMENT_PATTERNS:
        return {"error": "UNKNOWN_DOC_TYPE", "document_id": document_id}

    text_lower = raw_text.lower()
    extracted: dict = {}

    # Simulate extraction based on doc type
    if doc_type == "pay_stub":
        # Extract dollar amounts (e.g., $5,200.00)
        amounts = re.findall(r"\$?([\d,]+\.?\d*)", raw_text)
        numeric = [float(a.replace(",", "")) for a in amounts if a.replace(",", "").replace(".", "").isdigit()]
        extracted = {
            "employer_name": "Extracted Employer Inc.",
            "pay_period_end": (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d"),
            "gross_pay": max(numeric) if numeric else 0,
            "net_pay": min(numeric) if numeric else 0,
            "ytd_gross": (max(numeric) if numeric else 0) * random.randint(4, 11),
            "pay_frequency": "bi-weekly",
        }

    elif doc_type == "w2":
        amounts = re.findall(r"[\d,]+\.?\d*", raw_text)
        numeric = [float(a.replace(",", "")) for a in amounts if float(a.replace(",", "")) > 1000]
        extracted = {
            "tax_year": str(datetime.utcnow().year - 1),
            "employer_name": "Extracted Employer Inc.",
            "employer_ein": "XX-XXXXXXX",
            "box1_wages": max(numeric) if numeric else 0,
            "box2_federal_tax": (max(numeric) if numeric else 0) * 0.22,
            "box16_state_wages": max(numeric) if numeric else 0,
        }

    elif doc_type == "bank_statement":
        amounts = re.findall(r"[\d,]+\.?\d*", raw_text)
        numeric = sorted([float(a.replace(",", "")) for a in amounts if float(a.replace(",", "")) > 100], reverse=True)
        extracted = {
            "bank_name": "First National Bank",
            "account_last4": "1234",
            "statement_date": (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "ending_balance": numeric[0] if numeric else 0,
            "average_daily_balance": (numeric[0] if numeric else 0) * 0.85,
            "nsf_occurrences_90d": random.randint(0, 1),
        }

    elif doc_type == "tax_return_1040":
        amounts = re.findall(r"[\d,]+\.?\d*", raw_text)
        numeric = sorted([float(a.replace(",", "")) for a in amounts if float(a.replace(",", "")) > 10000], reverse=True)
        extracted = {
            "tax_year": str(datetime.utcnow().year - 1),
            "filing_status": "married_filing_jointly",
            "agi": numeric[0] if numeric else 0,
            "total_income": (numeric[0] if numeric else 0) * 1.05,
            "self_employment_income": 0,
        }

    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "extracted_data": extracted,
        "extraction_confidence": random.uniform(0.82, 0.98),
        "requires_manual_review": False,
        "extracted_at": datetime.utcnow().isoformat(),
    }


@tool("validate_document", args_schema=ValidateDocumentInput, return_direct=False)
def validate_document(
    document_id: str,
    doc_type: str,
    extracted_data: dict,
    borrower_name: str,
    annual_income_declared: float,
) -> dict:
    """Validate extracted document data for completeness, recency, and
    consistency with the borrower's declared income (FR-032, FR-033).
    Flags discrepancies for human review.
    """
    issues: list[str] = []
    warnings: list[str] = []

    pattern = DOCUMENT_PATTERNS.get(doc_type, {})
    required = pattern.get("required_fields", [])

    # Check required fields
    for field in required:
        if field not in extracted_data or not extracted_data[field]:
            issues.append(f"Missing required field: {field}")

    # Income consistency check
    if doc_type in ("pay_stub", "w2", "tax_return_1040"):
        doc_income = extracted_data.get("gross_pay", 0) or extracted_data.get("box1_wages", 0) or extracted_data.get("agi", 0)
        if doc_type == "pay_stub":
            doc_annual = doc_income * 26  # bi-weekly
        else:
            doc_annual = doc_income

        if doc_annual > 0 and annual_income_declared > 0:
            variance = abs(doc_annual - annual_income_declared) / annual_income_declared
            if variance > 0.20:
                issues.append(
                    f"Income variance exceeds 20%: declared ${annual_income_declared:,.0f} vs. "
                    f"document ${doc_annual:,.0f}"
                )
            elif variance > 0.10:
                warnings.append(f"Minor income variance ({variance:.1%}) — verify with borrower")

    # Document age check
    validity_days = pattern.get("validity_days")
    if validity_days:
        doc_date_str = extracted_data.get("pay_period_end") or extracted_data.get("statement_date")
        if doc_date_str:
            try:
                doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d")
                age_days = (datetime.utcnow() - doc_date).days
                if age_days > validity_days:
                    issues.append(
                        f"Document is {age_days} days old — exceeds {validity_days}-day freshness requirement"
                    )
            except ValueError:
                warnings.append("Could not parse document date — manual date verification required")

    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "action_required": "resubmit" if issues else ("review" if warnings else "none"),
        "validated_at": datetime.utcnow().isoformat(),
    }
