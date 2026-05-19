"""Document Agent — Stage 4 specialist (BRD §3.4, FR-030 through FR-034).

Orchestrates:
  • OCR document classification (ML-based)
  • Key data extraction from financial documents
  • Consistency validation against declared income
  • Missing document identification and re-request
  • Tamper-evident vault storage reference

LangSmith traces every call automatically.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import settings
from src.tools import (
    classify_document,
    extract_document_data,
    validate_document,
)


REQUIRED_DOCS_BY_PROFILE = {
    "employed": ["pay_stub", "pay_stub", "w2", "w2", "bank_statement", "bank_statement", "government_id"],
    "self_employed": ["tax_return_1040", "tax_return_1040", "bank_statement", "bank_statement", "government_id"],
    "retired": ["tax_return_1040", "bank_statement", "bank_statement", "government_id"],
}

DOC_DISPLAY_NAMES = {
    "pay_stub": "Pay Stub (2 most recent)",
    "w2": "W-2 Form (last 2 years)",
    "tax_return_1040": "Federal Tax Return 1040 (last 2 years)",
    "bank_statement": "Bank Statement (last 2 months)",
    "government_id": "Government-issued Photo ID",
    "purchase_contract": "Purchase Contract (for purchase loans)",
}

DOCUMENT_SYSTEM = """You are a mortgage document specialist. Your job is to help borrowers upload,
classify, and verify their financial documents efficiently and accurately.

For each document:
1. Classify it using classify_document
2. Extract key data using extract_document_data
3. Validate it using validate_document (checks freshness, completeness, income consistency)
4. Provide clear, plain-language feedback on what was found and any issues

Be specific about what's wrong with rejected documents — never say just "invalid".
If income on a document doesn't match what the borrower declared, explain the discrepancy
clearly and ask them to confirm or upload a corrected document.

Keep track of which required documents have been verified vs. are still outstanding.
"""


def _build_llm():
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.primary_llm_model, api_key=settings.anthropic_api_key, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o", temperature=0)


class DocumentAgent:
    """Manages the document collection and verification workflow."""

    def __init__(self):
        self._llm = _build_llm()
        self._tools = [classify_document, extract_document_data, validate_document]
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._tool_map = {t.name: t for t in self._tools}

    def get_required_documents(
        self,
        employment_status: str,
        loan_purpose: str,
    ) -> list[str]:
        """Return list of required document types for this borrower profile."""
        profile_key = employment_status if employment_status in REQUIRED_DOCS_BY_PROFILE else "employed"
        required = list(set(REQUIRED_DOCS_BY_PROFILE[profile_key]))
        if loan_purpose == "purchase":
            required.append("purchase_contract")
        return required

    def process_document(
        self,
        document_id: str,
        filename: str,
        file_size_kb: float,
        raw_text_excerpt: str,
        full_text: str,
        borrower_name: str,
        annual_income_declared: float,
    ) -> dict:
        """Process a single uploaded document through the full pipeline.

        Returns:
            {
                document_id, doc_type, verified, issues, warnings,
                extracted_data, feedback_message, action_required
            }
        """
        # Step 1: Classify
        classification = classify_document.invoke({
            "filename": filename,
            "file_size_kb": file_size_kb,
            "raw_text_excerpt": raw_text_excerpt,
        })

        if not classification.get("classified"):
            return {
                "document_id": document_id,
                "verified": False,
                "rejection_reason": classification.get("rejection_reason", "CLASSIFICATION_FAILED"),
                "feedback_message": classification.get("message", "Could not classify document."),
                "action_required": classification.get("action_required", "resubmit"),
            }

        doc_type = classification["doc_type"]

        # Step 2: Extract data
        extraction = extract_document_data.invoke({
            "document_id": document_id,
            "doc_type": doc_type,
            "raw_text": full_text,
        })

        extracted_data = extraction.get("extracted_data", {})

        # Step 3: Validate
        validation = validate_document.invoke({
            "document_id": document_id,
            "doc_type": doc_type,
            "extracted_data": extracted_data,
            "borrower_name": borrower_name,
            "annual_income_declared": annual_income_declared,
        })

        # Build feedback message
        if validation["valid"]:
            feedback = (
                f"Your {DOC_DISPLAY_NAMES.get(doc_type, doc_type)} has been accepted and verified."
            )
            if validation.get("warnings"):
                feedback += f" Note: {'; '.join(validation['warnings'])}"
        else:
            issues = "; ".join(validation.get("issues", ["Unknown issue"]))
            feedback = (
                f"There's an issue with your {DOC_DISPLAY_NAMES.get(doc_type, doc_type)}: "
                f"{issues}. Please upload a corrected document."
            )

        return {
            "document_id": document_id,
            "doc_type": doc_type,
            "filename": filename,
            "classified": True,
            "classification_confidence": classification.get("confidence", 0),
            "extracted_data": extracted_data,
            "verified": validation["valid"],
            "issues": validation.get("issues", []),
            "warnings": validation.get("warnings", []),
            "feedback_message": feedback,
            "action_required": validation.get("action_required", "none"),
            "processed_at": datetime.utcnow().isoformat(),
        }

    def get_document_status(
        self,
        documents_processed: list[dict],
        employment_status: str,
        loan_purpose: str,
    ) -> dict:
        """Summarize what documents have been received vs. what's still missing."""
        required = self.get_required_documents(employment_status, loan_purpose)
        verified_types = [d["doc_type"] for d in documents_processed if d.get("verified")]

        missing = []
        for req_type in required:
            count_needed = required.count(req_type)
            count_have = verified_types.count(req_type)
            if count_have < count_needed:
                missing.append(DOC_DISPLAY_NAMES.get(req_type, req_type))

        # Deduplicate
        missing = list(dict.fromkeys(missing))

        return {
            "total_required": len(set(required)),
            "verified_count": len(set(verified_types)),
            "missing_documents": missing,
            "all_documents_received": len(missing) == 0,
            "summary": (
                f"{len(set(verified_types))} of {len(set(required))} document types verified. "
                + (f"Still needed: {', '.join(missing)}" if missing else "All documents received!")
            ),
        }

    def generate_request_message(self, missing_docs: list[str]) -> str:
        """Generate a friendly document request message."""
        if not missing_docs:
            return "All required documents have been received and verified. You're ready to proceed!"

        lines = ["We still need the following documents to complete your application:\n"]
        for i, doc in enumerate(missing_docs, 1):
            lines.append(f"  {i}. {doc}")
        lines.append(
            "\nYou can upload them by taking a photo with your phone or uploading a file from your computer. "
            "Accepted formats: PDF, JPG, PNG (max 25 MB per file)."
        )
        return "\n".join(lines)
