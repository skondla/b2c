"""Document Extraction Skill — bulk document pipeline (FR-031, FR-032).

Composable skill that processes multiple documents in sequence,
returning a unified verification report suitable for the underwriter.
"""

from __future__ import annotations

from datetime import datetime

from src.agents.document_agent import DocumentAgent


class DocumentExtractionSkill:
    """Bulk document extraction and verification pipeline."""

    def __init__(self):
        self._agent = DocumentAgent()

    def process_batch(
        self,
        documents: list[dict],
        borrower_name: str,
        annual_income_declared: float,
        employment_status: str = "employed",
        loan_purpose: str = "purchase",
    ) -> dict:
        """Process a batch of uploaded documents.

        Args:
            documents: List of {document_id, filename, file_size_kb, raw_text_excerpt, full_text}
            borrower_name: For identity validation
            annual_income_declared: For income consistency checks
            employment_status: 'employed', 'self_employed', 'retired'
            loan_purpose: 'purchase' or 'refinance'

        Returns:
            {
                processed: [doc results],
                verified_count: int,
                failed_count: int,
                status: dict from get_document_status,
                verification_report: str (for underwriter)
            }
        """
        processed = []
        for doc in documents:
            result = self._agent.process_document(
                document_id=doc.get("document_id", ""),
                filename=doc.get("filename", ""),
                file_size_kb=doc.get("file_size_kb", 0),
                raw_text_excerpt=doc.get("raw_text_excerpt", "")[:500],
                full_text=doc.get("full_text", ""),
                borrower_name=borrower_name,
                annual_income_declared=annual_income_declared,
            )
            processed.append(result)

        verified = [d for d in processed if d.get("verified")]
        failed = [d for d in processed if not d.get("verified")]

        status = self._agent.get_document_status(
            processed, employment_status, loan_purpose
        )

        # Build underwriter-facing verification report
        report_lines = [
            f"DOCUMENT VERIFICATION REPORT",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Borrower: {borrower_name}",
            f"",
            f"SUMMARY: {len(verified)} verified / {len(processed)} submitted",
            f"",
        ]
        for doc in processed:
            icon = "✓" if doc.get("verified") else "✗"
            report_lines.append(
                f"{icon} {doc.get('doc_type', 'unknown')} — {doc.get('filename', '')}"
            )
            if doc.get("issues"):
                for issue in doc["issues"]:
                    report_lines.append(f"    ISSUE: {issue}")
            if doc.get("warnings"):
                for warn in doc["warnings"]:
                    report_lines.append(f"    WARN: {warn}")
            if doc.get("extracted_data"):
                data = doc["extracted_data"]
                if "gross_pay" in data:
                    report_lines.append(f"    Gross Pay: ${data['gross_pay']:,.0f}")
                if "ending_balance" in data:
                    report_lines.append(f"    Bank Balance: ${data['ending_balance']:,.0f}")

        if status.get("missing_documents"):
            report_lines.append(f"\nSTILL OUTSTANDING:")
            for doc in status["missing_documents"]:
                report_lines.append(f"  - {doc}")

        return {
            "processed": processed,
            "verified_count": len(verified),
            "failed_count": len(failed),
            "all_verified": status.get("all_documents_received", False),
            "missing_documents": status.get("missing_documents", []),
            "status_summary": status.get("summary", ""),
            "borrower_message": self._agent.generate_request_message(
                status.get("missing_documents", [])
            ),
            "verification_report": "\n".join(report_lines),
        }
