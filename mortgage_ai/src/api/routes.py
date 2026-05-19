"""API routes for the B2C Mortgage AI Platform.

Endpoints:
  POST /applications                  — Create application + start workflow
  GET  /applications/{id}             — Get application state
  POST /applications/{id}/chat        — Send message, get AI response
  GET  /applications/{id}/stream      — SSE streaming chat
  POST /applications/{id}/prequal     — Run pre-qualification skill
  POST /applications/{id}/documents   — Process uploaded document
  GET  /applications/{id}/documents/status  — Document status
  POST /applications/{id}/rate-lock   — Lock interest rate
  POST /applications/{id}/disclosures — Deliver disclosures
  GET  /knowledge/search              — Semantic knowledge search
  POST /lo/analyze                    — Loan Officer copilot: analyze file
  POST /lo/draft-email                — LO copilot: draft borrower email
  GET  /lo/compliance-calendar/{id}   — LO compliance deadlines
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.graphs.mortgage_graph import run_mortgage_workflow, stream_mortgage_workflow
from src.graphs.states import LoanStage, default_state
from src.skills.prequalification_skill import PrequalificationSkill
from src.skills.document_extraction_skill import DocumentExtractionSkill
from src.agents.loan_officer_agent import LoanOfficerCopilot
from src.tools.knowledge_tools import search_mortgage_knowledge
from src.tools.rate_tools import lock_rate

router = APIRouter()

# ── In-memory session store (replace with Redis/DB in production) ─────────────
_sessions: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class CreateApplicationRequest(BaseModel):
    borrower_name: str
    borrower_email: str
    borrower_phone: str = ""
    loan_purpose: str = "purchase"
    property_type: str = "single_family"


class ChatRequest(BaseModel):
    message: str = Field(description="Borrower's message or response")


class PrequalRequest(BaseModel):
    annual_income: float
    monthly_debt_payments: float = 0.0
    estimated_home_price: float
    down_payment: float
    credit_score_range_min: int = Field(description="Lower bound of borrower's estimated credit score")
    loan_purpose: str = "purchase"
    property_type: str = "single_family"
    veteran_status: bool = False


class RateLockRequest(BaseModel):
    loan_product: str
    rate_pct: float
    lock_period_days: int = 30


class DisclosureRequest(BaseModel):
    disclosure_type: str = Field(description="loan_estimate or closing_disclosure")
    ip_address: str
    user_agent: str = ""


class LOAnalyzeRequest(BaseModel):
    application_id: str
    lo_question: str = ""
    lo_id: str = ""


class LODraftEmailRequest(BaseModel):
    application_id: str
    communication_type: str
    context: dict = Field(default_factory=dict)
    lo_name: str = "Your Loan Officer"
    lo_email: str = "loans@example.com"
    lo_phone: str = "1-800-MORTGAGE"


# ── Application lifecycle ─────────────────────────────────────────────────────

@router.post("/applications", tags=["Applications"], status_code=201)
async def create_application(req: CreateApplicationRequest):
    """Create a new mortgage application and initialize the workflow (BRD §3.1)."""
    application_id = str(uuid.uuid4())
    borrower_id = str(uuid.uuid4())

    state = default_state(
        application_id=application_id,
        borrower_id=borrower_id,
        borrower_name=req.borrower_name,
        borrower_email=req.borrower_email,
    )
    state["borrower_phone"] = req.borrower_phone
    state["loan_purpose"] = req.loan_purpose
    state["property_type"] = req.property_type

    # Run discovery stage
    updated_state, response = run_mortgage_workflow(
        state=state,
        user_message="Start my mortgage application",
        thread_id=application_id,
    )

    _sessions[application_id] = updated_state

    return {
        "application_id": application_id,
        "borrower_id": borrower_id,
        "stage": updated_state.get("stage"),
        "welcome_message": response,
        "created_at": state["created_at"],
        "_links": {
            "self": f"/api/v1/applications/{application_id}",
            "chat": f"/api/v1/applications/{application_id}/chat",
            "stream": f"/api/v1/applications/{application_id}/stream",
            "prequal": f"/api/v1/applications/{application_id}/prequal",
        },
    }


@router.get("/applications/{application_id}", tags=["Applications"])
async def get_application(application_id: str):
    """Retrieve current application state and journey progress."""
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    return {
        "application_id": application_id,
        "stage": state.get("stage"),
        "borrower_name": state.get("borrower_name"),
        "loan_purpose": state.get("loan_purpose"),
        "loan_amount": state.get("loan_amount", 0),
        "prequal_eligible": state.get("prequal_eligible"),
        "urla_complete": state.get("urla_complete", False),
        "documents_verified": state.get("documents_verified", False),
        "rate_locked": state.get("rate_locked", False),
        "e_consent_given": state.get("e_consent_given", False),
        "aus_decision": state.get("aus_decision"),
        "conditions": state.get("conditions", []),
        "compliance_flags": state.get("compliance_flags", []),
        "human_review_required": state.get("human_review_required", False),
        "created_at": state.get("created_at"),
    }


# ── Conversational chat ───────────────────────────────────────────────────────

@router.post("/applications/{application_id}/chat", tags=["Chat"])
async def chat(application_id: str, req: ChatRequest):
    """Send a message and get an AI response (non-streaming).
    The graph automatically routes to the correct stage node.
    """
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    updated_state, response = run_mortgage_workflow(
        state=state,
        user_message=req.message,
        thread_id=application_id,
    )
    _sessions[application_id] = updated_state

    return {
        "application_id": application_id,
        "stage": updated_state.get("stage"),
        "response": response,
        "human_review_required": updated_state.get("human_review_required", False),
        "compliance_flags": updated_state.get("compliance_flags", []),
    }


@router.get("/applications/{application_id}/stream", tags=["Chat"])
async def stream_chat(
    application_id: str,
    message: str = Query(description="Borrower's message"),
):
    """SSE streaming endpoint — returns AI response token by token.
    Supports real-time status updates and stage transitions.
    """
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    async def event_generator():
        for event_type, data in stream_mortgage_workflow(
            state=state,
            user_message=message,
            thread_id=application_id,
        ):
            import json
            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Pre-qualification ─────────────────────────────────────────────────────────

@router.post("/applications/{application_id}/prequal", tags=["Pre-Qualification"])
async def run_prequal(application_id: str, req: PrequalRequest):
    """Run pre-qualification and return result within 60 seconds (NFR-002, FR-010).
    Performs soft credit check — no impact to borrower's credit score.
    """
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    skill = PrequalificationSkill()
    result = skill.run(
        annual_income=req.annual_income,
        monthly_debt_payments=req.monthly_debt_payments,
        estimated_home_price=req.estimated_home_price,
        down_payment=req.down_payment,
        credit_score_range_min=req.credit_score_range_min,
        loan_purpose=req.loan_purpose,
        property_type=req.property_type,
        veteran_status=req.veteran_status,
        borrower_name=state.get("borrower_name", "Borrower"),
    )

    # Update application state
    state.update({
        "annual_income": req.annual_income,
        "monthly_debt_payments": req.monthly_debt_payments,
        "estimated_home_price": req.estimated_home_price,
        "down_payment": req.down_payment,
        "loan_amount": req.estimated_home_price - req.down_payment,
        "credit_score_range_min": req.credit_score_range_min,
        "loan_purpose": req.loan_purpose,
        "property_type": req.property_type,
        "veteran_status": req.veteran_status,
        "prequal_eligible": result.get("eligible", False),
        "prequal_eligible_products": result.get("eligible_products", []),
        "prequal_letter_url": result.get("prequal_letter_url"),
        "prequal_completed_at": datetime.utcnow().isoformat(),
        "stage": LoanStage.APPLICATION.value if result.get("eligible") else LoanStage.PREQUALIFICATION.value,
    })
    _sessions[application_id] = state

    return {
        "application_id": application_id,
        **result,
    }


# ── Document processing ───────────────────────────────────────────────────────

@router.post("/applications/{application_id}/documents", tags=["Documents"])
async def upload_document(
    application_id: str,
    filename: str = Form(...),
    file_size_kb: float = Form(...),
    raw_text_excerpt: str = Form(...),
    full_text: str = Form(default=""),
):
    """Upload and process a document (FR-030, FR-031, FR-032).
    Accepts text content extracted from PDF/image (in production: raw binary + server-side OCR).
    Returns classification, extraction, and validation results immediately.
    """
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    if file_size_kb > settings.doc_upload_max_mb * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.doc_upload_max_mb} MB limit")

    skill = DocumentExtractionSkill()
    document_id = str(uuid.uuid4())

    result = skill._agent.process_document(
        document_id=document_id,
        filename=filename,
        file_size_kb=file_size_kb,
        raw_text_excerpt=raw_text_excerpt[:500],
        full_text=full_text or raw_text_excerpt,
        borrower_name=state.get("borrower_name", "Borrower"),
        annual_income_declared=state.get("annual_income", 0),
    )

    # Add to application documents list
    docs = state.get("documents_uploaded", [])
    docs.append({
        "document_id": document_id,
        "doc_type": result.get("doc_type", "unknown"),
        "filename": filename,
        "verified": result.get("verified", False),
        "uploaded_at": datetime.utcnow().isoformat(),
    })
    state["documents_uploaded"] = docs
    _sessions[application_id] = state

    return {
        "application_id": application_id,
        "document_id": document_id,
        **result,
    }


@router.get("/applications/{application_id}/documents/status", tags=["Documents"])
async def get_document_status(application_id: str):
    """Get current document collection status (FR-034)."""
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    skill = DocumentExtractionSkill()
    status = skill._agent.get_document_status(
        documents_processed=state.get("documents_uploaded", []),
        employment_status=state.get("employment_status", "employed"),
        loan_purpose=state.get("loan_purpose", "purchase"),
    )

    return {"application_id": application_id, **status}


# ── Rate lock ─────────────────────────────────────────────────────────────────

@router.post("/applications/{application_id}/rate-lock", tags=["Loan Selection"])
async def lock_rate_endpoint(application_id: str, req: RateLockRequest):
    """Lock interest rate for the application (BRD §3.5)."""
    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    result = lock_rate.invoke({
        "application_id": application_id,
        "loan_amount": state.get("loan_amount", 0),
        "loan_product": req.loan_product,
        "rate_pct": req.rate_pct,
        "lock_period_days": req.lock_period_days,
        "borrower_name": state.get("borrower_name", ""),
    })

    if "error" not in result:
        state.update({
            "rate_locked": True,
            "rate_lock_id": result.get("lock_id"),
            "selected_product": req.loan_product,
            "interest_rate": result.get("locked_rate_pct"),
        })
        _sessions[application_id] = state

    return {"application_id": application_id, **result}


# ── Disclosures ───────────────────────────────────────────────────────────────

@router.post("/applications/{application_id}/disclosures", tags=["Disclosures"])
async def deliver_disclosure(application_id: str, req: DisclosureRequest):
    """Deliver a TRID-compliant disclosure and capture e-signature intent (FR-040 through FR-043)."""
    from src.tools.compliance_tools import check_trid_compliance, capture_esign_consent

    state = _sessions.get(application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    # Capture e-consent first if not already done (FR-041)
    if not state.get("e_consent_given"):
        consent = capture_esign_consent.invoke({
            "application_id": application_id,
            "borrower_id": state.get("borrower_id", ""),
            "ip_address": req.ip_address,
            "user_agent": req.user_agent,
        })
        state["e_consent_given"] = True
        state["e_consent_id"] = consent.get("consent_id")

    # Check TRID compliance
    trid = check_trid_compliance.invoke({
        "application_id": application_id,
        "disclosure_type": req.disclosure_type,
        "application_received_at": state.get("created_at", datetime.utcnow().isoformat()),
        "scheduled_closing_date": state.get("closing_date", "2026-09-01"),
        "disclosures_delivered": state.get("disclosures_delivered", []),
    })

    if not trid.get("compliant") and trid.get("issues"):
        # Flag but don't block — log for compliance team
        flags = state.get("compliance_flags", [])
        flags.extend(trid.get("issues", []))
        state["compliance_flags"] = flags

    # Record delivery
    delivered = state.get("disclosures_delivered", [])
    delivered.append(req.disclosure_type)
    state["disclosures_delivered"] = delivered
    _sessions[application_id] = state

    return {
        "application_id": application_id,
        "disclosure_type": req.disclosure_type,
        "delivered_at": datetime.utcnow().isoformat(),
        "ip_address": req.ip_address,
        "trid_compliance": trid,
        "e_consent_id": state.get("e_consent_id"),
        "audit_trail": {
            "delivered_at": datetime.utcnow().isoformat(),
            "ip": req.ip_address,
            "user_agent": req.user_agent,
        },
    }


# ── Knowledge base search ─────────────────────────────────────────────────────

@router.get("/knowledge/search", tags=["Knowledge"])
async def knowledge_search(
    q: str = Query(description="Search query"),
    n: int = Query(default=3, ge=1, le=5),
    category: str = Query(default="", description="products, regulations, faq, or empty"),
):
    """Semantic vector search over mortgage knowledge base (FR-054)."""
    result = search_mortgage_knowledge.invoke({
        "query": q,
        "n_results": n,
        "filter_category": category,
    })
    return result


# ── Loan Officer Copilot ──────────────────────────────────────────────────────

@router.post("/lo/analyze", tags=["Loan Officer"])
async def lo_analyze(req: LOAnalyzeRequest):
    """Loan Officer copilot: AI-powered application file analysis (FR-024, FR-052)."""
    state = _sessions.get(req.application_id)
    if not state:
        raise HTTPException(status_code=404, detail="Application not found")

    copilot = LoanOfficerCopilot()
    analysis = copilot.analyze_application(state, req.lo_question)
    compliance_alerts = copilot.check_compliance_calendar(state)

    return {
        "application_id": req.application_id,
        "analysis": analysis,
        "compliance_alerts": compliance_alerts,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/lo/draft-email", tags=["Loan Officer"])
async def lo_draft_email(req: LODraftEmailRequest):
    """Loan Officer copilot: draft borrower communications (FR-052)."""
    state = _sessions.get(req.application_id, {})
    borrower_name = state.get("borrower_name", "Borrower")

    copilot = LoanOfficerCopilot()
    email_draft = copilot.draft_borrower_communication(
        communication_type=req.communication_type,
        borrower_name=borrower_name,
        context=req.context,
        lo_name=req.lo_name,
        lo_email=req.lo_email,
        lo_phone=req.lo_phone,
    )

    return {
        "application_id": req.application_id,
        "communication_type": req.communication_type,
        "draft": email_draft,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/lo/compliance-calendar/{application_id}", tags=["Loan Officer"])
async def lo_compliance_calendar(application_id: str):
    """Loan Officer: get upcoming regulatory deadlines (FR-040 through FR-044)."""
    state = _sessions.get(application_id, {})
    copilot = LoanOfficerCopilot()
    alerts = copilot.check_compliance_calendar(state)

    return {
        "application_id": application_id,
        "compliance_alerts": alerts,
        "as_of": datetime.utcnow().isoformat(),
    }
