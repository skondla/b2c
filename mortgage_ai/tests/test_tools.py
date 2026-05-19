"""Unit tests for LangChain tools — no external API calls required."""

import pytest
from src.tools.credit_tools import soft_credit_check, hard_credit_check
from src.tools.eligibility_tools import check_eligibility, calculate_prequal
from src.tools.document_tools import classify_document, validate_document
from src.tools.rate_tools import calculate_payment, calculate_apr, lock_rate
from src.tools.compliance_tools import check_trid_compliance, check_hmda_data


# ── Credit tools ──────────────────────────────────────────────────────────────

class TestCreditTools:
    def test_soft_credit_check_returns_score_range(self):
        result = soft_credit_check.invoke({
            "borrower_name": "Alex Johnson",
            "ssn_last4": "1234",
            "annual_income": 120_000,
            "date_of_birth": "1990-06-15",
        })
        assert result["inquiry_type"] == "soft"
        assert "score_range" in result
        assert result["score_impact"] == "none — soft inquiry only"
        assert result["eligible_for_prequal"] in (True, False)

    def test_hard_credit_check_requires_consent(self):
        result = hard_credit_check.invoke({
            "borrower_name": "Jordan Smith",
            "ssn_last4": "5678",
            "annual_income": 95_000,
            "date_of_birth": "1985-03-20",
            "application_id": "test-123",
            "consented": False,
        })
        assert result["error"] == "FCRA_CONSENT_REQUIRED"

    def test_hard_credit_check_with_consent(self):
        result = hard_credit_check.invoke({
            "borrower_name": "Jordan Smith",
            "ssn_last4": "5678",
            "annual_income": 95_000,
            "date_of_birth": "1985-03-20",
            "application_id": "test-123",
            "consented": True,
        })
        assert "fico_score" in result
        assert "credit_report_id" in result
        assert result["credit_report_id"].startswith("CR-")
        assert len(result["bureaus_queried"]) == 3


# ── Eligibility tools ─────────────────────────────────────────────────────────

class TestEligibilityTools:
    def test_eligible_conventional_borrower(self):
        result = check_eligibility.invoke({
            "estimated_home_price": 400_000,
            "down_payment": 80_000,  # 20% down
            "annual_income": 120_000,
            "monthly_debt_payments": 500,
            "credit_score": 720,
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "veteran_status": False,
        })
        assert result["eligible"] is True
        assert "conventional_30_fixed" in result["eligible_products"]
        assert result["ltv_ratio"] == pytest.approx(0.80, rel=0.01)

    def test_ineligible_low_credit_score(self):
        result = check_eligibility.invoke({
            "estimated_home_price": 300_000,
            "down_payment": 10_000,
            "annual_income": 80_000,
            "monthly_debt_payments": 1_000,
            "credit_score": 490,  # below all minimums
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "veteran_status": False,
        })
        assert result["eligible"] is False
        assert len(result["ineligibility_reasons"]) > 0

    def test_va_loan_only_for_veterans(self):
        result_non_vet = check_eligibility.invoke({
            "estimated_home_price": 500_000,
            "down_payment": 0,
            "annual_income": 100_000,
            "monthly_debt_payments": 300,
            "credit_score": 700,
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "veteran_status": False,
        })
        result_vet = check_eligibility.invoke({
            "estimated_home_price": 500_000,
            "down_payment": 0,
            "annual_income": 100_000,
            "monthly_debt_payments": 300,
            "credit_score": 700,
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "veteran_status": True,
        })
        assert "va_30_fixed" not in result_non_vet["eligible_products"]
        assert "va_30_fixed" in result_vet["eligible_products"]

    def test_prequal_returns_letter_url(self):
        result = calculate_prequal.invoke({
            "estimated_home_price": 350_000,
            "down_payment": 70_000,
            "annual_income": 100_000,
            "monthly_debt_payments": 400,
            "credit_score_range_min": 700,
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "veteran_status": False,
        })
        assert result["eligible"] is True
        assert result["prequal_letter_id"].startswith("PQL-")
        assert "/documents/prequal/" in result["letter_url"]


# ── Document tools ────────────────────────────────────────────────────────────

class TestDocumentTools:
    def test_classify_pay_stub(self):
        result = classify_document.invoke({
            "filename": "paystub_may2026.pdf",
            "file_size_kb": 150.0,
            "raw_text_excerpt": "Pay Period: May 1-15 2026  Gross Pay: $4,600  Net Pay: $3,200  YTD Gross: $41,400  Employer: Acme Corp",
        })
        assert result["classified"] is True
        assert result["doc_type"] == "pay_stub"

    def test_classify_bank_statement(self):
        result = classify_document.invoke({
            "filename": "bank_april.pdf",
            "file_size_kb": 200.0,
            "raw_text_excerpt": "Account number ending 1234  Beginning Balance: $12,000  Ending Balance: $15,500  Routing: 021000021",
        })
        assert result["classified"] is True
        assert result["doc_type"] == "bank_statement"

    def test_illegible_document(self):
        result = classify_document.invoke({
            "filename": "blurry_scan.jpg",
            "file_size_kb": 500.0,
            "raw_text_excerpt": "   ",
        })
        assert result["classified"] is False
        assert result["rejection_reason"] == "ILLEGIBLE"

    def test_file_too_large(self):
        result = classify_document.invoke({
            "filename": "huge.pdf",
            "file_size_kb": 30_000.0,  # 30 MB
            "raw_text_excerpt": "Some content here",
        })
        assert "error" in result
        assert result["error"] == "FILE_TOO_LARGE"

    def test_validate_income_consistency(self):
        # Document shows much higher income than declared
        result = validate_document.invoke({
            "document_id": "doc-001",
            "doc_type": "pay_stub",
            "extracted_data": {
                "employer_name": "BigCo",
                "pay_period_end": "2026-05-01",
                "gross_pay": 10_000,  # $260k annualized vs $80k declared
                "net_pay": 7_000,
                "ytd_gross": 45_000,
            },
            "borrower_name": "Test Borrower",
            "annual_income_declared": 80_000,
        })
        # Should flag income variance
        assert len(result["issues"]) > 0 or len(result["warnings"]) > 0


# ── Rate tools ────────────────────────────────────────────────────────────────

class TestRateTools:
    def test_calculate_payment_30yr(self):
        result = calculate_payment.invoke({
            "loan_amount": 300_000,
            "annual_rate_pct": 6.875,
            "term_years": 30,
            "loan_product": "conventional_30_fixed",
            "down_payment": 60_000,
            "home_value": 360_000,
        })
        # Standard 30yr at 6.875% should be ~$1,970/mo P&I
        assert 1_900 < result["monthly_pi"] < 2_100
        assert result["total_closing_costs"] > 5_000

    def test_apr_higher_than_rate(self):
        result = calculate_apr.invoke({
            "loan_amount": 300_000,
            "annual_rate_pct": 6.875,
            "term_years": 30,
            "origination_fee_pct": 0.01,
            "discount_points": 0.0,
            "other_fees": 1_500.0,
        })
        assert result["apr_pct"] > result["note_rate_pct"]

    def test_rate_lock_valid_period(self):
        result = lock_rate.invoke({
            "application_id": "app-001",
            "loan_amount": 300_000,
            "loan_product": "conventional_30_fixed",
            "rate_pct": 6.875,
            "lock_period_days": 30,
            "borrower_name": "Test Borrower",
        })
        assert result["lock_id"].startswith("LCK-")
        assert result["lock_period_days"] == 30

    def test_rate_lock_invalid_period(self):
        result = lock_rate.invoke({
            "application_id": "app-001",
            "loan_amount": 300_000,
            "loan_product": "conventional_30_fixed",
            "rate_pct": 6.875,
            "lock_period_days": 90,  # invalid
            "borrower_name": "Test Borrower",
        })
        assert "error" in result


# ── Compliance tools ──────────────────────────────────────────────────────────

class TestComplianceTools:
    def test_hmda_complete(self):
        result = check_hmda_data.invoke({
            "application_id": "app-001",
            "loan_type": "conventional",
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "occupancy_type": "principal_residence",
            "loan_amount": 350_000,
            "property_address": "123 Main St, Austin, TX 78701",
            "applicant_ethnicity": "not_hispanic",
            "applicant_race": "white",
            "applicant_sex": "male",
            "annual_income": 100_000,
            "opted_out": False,
        })
        assert result["hmda_complete"] is True
        assert result["lar_reportable"] is True

    def test_hmda_opt_out(self):
        result = check_hmda_data.invoke({
            "application_id": "app-001",
            "loan_type": "conventional",
            "loan_purpose": "purchase",
            "property_type": "single_family",
            "occupancy_type": "principal_residence",
            "loan_amount": 350_000,
            "property_address": "123 Main St, Austin, TX 78701",
            "applicant_ethnicity": "provided",
            "applicant_race": "provided",
            "applicant_sex": "provided",
            "annual_income": 100_000,
            "opted_out": True,
        })
        assert result["opted_out"] is True
        assert result["hmda_record"]["applicant_race"] == "information_not_provided"

    def test_trid_loan_estimate_timing(self):
        from datetime import datetime, timedelta
        # Application received today, LE not yet sent
        app_time = (datetime.utcnow() - timedelta(days=4)).isoformat()
        result = check_trid_compliance.invoke({
            "application_id": "app-001",
            "disclosure_type": "loan_estimate",
            "application_received_at": app_time,
            "scheduled_closing_date": "2026-08-01",
            "disclosures_delivered": [],
        })
        # 4 days past application → should have compliance issue
        assert len(result["issues"]) > 0 or not result["compliant"]
