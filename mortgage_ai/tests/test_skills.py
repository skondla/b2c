"""Tests for pre-qualification skill (no LLM calls — tests tool pipeline only)."""

import pytest
from unittest.mock import patch, MagicMock
from src.skills.prequalification_skill import PrequalificationSkill
from src.skills.document_extraction_skill import DocumentExtractionSkill


class TestPrequalificationSkill:

    def test_eligible_borrower(self):
        """Test full prequal pipeline with an eligible borrower profile."""
        skill = PrequalificationSkill()

        # Mock the LLM summary to avoid API calls in CI
        with patch.object(skill, '_summary_chain') as mock_chain:
            mock_chain.invoke.return_value = {
                "headline": "You pre-qualify for up to $320,000!",
                "eligible": True,
                "max_loan_formatted": "$320,000",
                "rate_range": "6.625% - 7.250%",
                "recommended_product_display": "30-Year Fixed Conventional",
                "monthly_payment_estimate": "$2,100/month",
                "dti_ratio": "38.5%",
                "ltv_ratio": "80.0%",
                "next_steps": ["Complete full application"],
                "ineligibility_reasons": [],
                "recommendations": [],
                "disclaimer": "Not a commitment to lend.",
            }

            result = skill.run(
                annual_income=100_000,
                monthly_debt_payments=500,
                estimated_home_price=400_000,
                down_payment=80_000,
                credit_score_range_min=720,
            )

        assert result["eligible"] is True
        assert result["prequal_letter_id"] is not None
        assert result["credit_score_range"] != ""

    def test_ineligible_borrower(self):
        """Test with profile that should not qualify."""
        skill = PrequalificationSkill()

        with patch.object(skill, '_summary_chain') as mock_chain:
            mock_chain.invoke.return_value = {
                "eligible": False,
                "headline": "Unable to pre-qualify at this time.",
                "ineligibility_reasons": ["Credit score too low"],
                "recommendations": ["Work on credit score"],
                "max_loan_formatted": "$0",
                "rate_range": "N/A",
                "recommended_product_display": "N/A",
                "monthly_payment_estimate": "N/A",
                "dti_ratio": "N/A",
                "ltv_ratio": "N/A",
                "next_steps": [],
                "disclaimer": "",
            }

            result = skill.run(
                annual_income=30_000,
                monthly_debt_payments=2_000,
                estimated_home_price=500_000,
                down_payment=5_000,
                credit_score_range_min=490,  # Below all minimums
            )

        # Raw prequal data from tools should show ineligible
        assert "raw_prequal" in result
        assert "raw_eligibility" in result


class TestDocumentExtractionSkill:

    def test_process_batch_all_valid(self):
        skill = DocumentExtractionSkill()
        docs = [
            {
                "document_id": "doc-001",
                "filename": "paystub.pdf",
                "file_size_kb": 150.0,
                "raw_text_excerpt": "Pay Period May 2026  Gross Pay $5000  Net Pay $3500  YTD Gross $45000  Employer Acme Corp",
                "full_text": "Pay Period May 2026  Gross Pay $5000  Net Pay $3500  YTD Gross $45000  Employer Acme Corp pay period end date 2026-05-15",
            },
        ]
        result = skill.process_batch(
            documents=docs,
            borrower_name="Alex Test",
            annual_income_declared=130_000,  # ~$5k/2wks * 26 = $130k
            employment_status="employed",
            loan_purpose="purchase",
        )

        assert "processed" in result
        assert "verification_report" in result
        assert len(result["processed"]) == 1

    def test_process_batch_empty(self):
        skill = DocumentExtractionSkill()
        result = skill.process_batch(
            documents=[],
            borrower_name="Test Borrower",
            annual_income_declared=100_000,
        )
        assert result["verified_count"] == 0
        assert result["all_verified"] is False
        assert len(result["missing_documents"]) > 0
