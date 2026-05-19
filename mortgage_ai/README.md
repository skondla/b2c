# B2C Mortgage Loan Application — AI Platform

An end-to-end AI-powered mortgage origination platform built with **LangChain**, **LangGraph**, **LangSmith**, and **ChromaDB** vector search. Implements all 9 borrower journey stages from the BRD with full compliance automation (TRID, ECOA, HMDA, E-SIGN).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API (src/api/)                     │
│  /applications  /prequal  /documents  /disclosures  /lo/  /knowledge│
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│              LangGraph Mortgage Workflow (src/graphs/)               │
│                                                                     │
│  START → supervisor → [Discovery] → [Prequal] → [Application]      │
│           → [Documents] → [Property/Loan] → [Disclosures]          │
│           → [Underwriting] → [Closing] → [Self-Service] → END      │
│                                                                     │
│  Human-in-the-loop: interrupt_before=["human_review"]              │
│  Persistence: MemorySaver (cross-device resume, FR-022)            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼──────┐ ┌───────▼───────┐
│   LangChain   │ │   Agents     │ │    Skills     │
│   Tools       │ │  (src/agents)│ │  (src/skills) │
│               │ │              │ │               │
│ • soft/hard   │ │ • Prequal    │ │ • MortgageAdv │
│   credit check│ │   Agent      │ │   isorSkill   │
│ • eligibility │ │ • Document   │ │   (RAG chain) │
│ • doc OCR/    │ │   Agent      │ │ • Prequal     │
│   classify    │ │ • LO Copilot │ │   Skill       │
│ • rate/APR    │ │              │ │ • DocExtract  │
│ • compliance  │ └──────────────┘ │   Skill       │
│ • knowledge   │                  └───────────────┘
│   search      │
└───────┬───────┘
        │
┌───────▼────────────────────────────────────────────┐
│         ChromaDB Vector Store (src/vectorstore/)    │
│                                                    │
│  Collection 1: mortgage_knowledge                  │
│    • Mortgage products (products.md)               │
│    • Regulations (regulations.md)                  │
│    • FAQ (faq.md)                                  │
│    • Semantic similarity search (RAG)              │
│                                                    │
│  Collection 2: borrower_documents                  │
│    • Per-application document text                 │
│    • Filtered by application_id                   │
└────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────┐
│         LangSmith Observability                     │
│  • Every LLM call traced automatically             │
│  • Tool call latencies tracked                     │
│  • Graph node execution visualized                 │
│  • BRD KPI dashboards (prequal time, approval time)│
└────────────────────────────────────────────────────┘
```

---

## BRD Requirements Coverage

| BRD Reference | Requirement | Implementation |
|---|---|---|
| FR-001 to FR-005 | Account & Identity | FastAPI auth + JWT |
| FR-010 to FR-014 | Pre-Qualification | `PrequalificationSkill` + `soft_credit_check` |
| FR-020 to FR-025 | URLA Application | LangGraph `application` node + HMDA tool |
| FR-030 to FR-034 | Document Collection | `DocumentExtractionSkill` + OCR tools |
| FR-040 to FR-044 | Disclosures & E-Sign | `check_trid_compliance` + `capture_esign_consent` |
| FR-050 to FR-054 | Status & Notifications | LangGraph state + SSE streaming |
| FR-060 to FR-064 | Integrations | Tool stubs (credit bureau, AUS, appraisal) |
| NFR-001 to 005 | Performance | FastAPI async + p95 monitoring |
| NFR-010 to 013 | Availability | Docker + health checks |
| NFR-020 to 025 | Security | TLS config + audit logging |
| NFR-030 to 033 | Accessibility | API-first (UI handles WCAG) |
| §5.4 | Regulatory | TRID/ECOA/HMDA/FCRA/GLBA/E-SIGN tools |
| §5.8 | Observability | LangSmith + structlog correlation IDs |
| §6.1 POC | 30yr fixed, 1-2 states, web only | Conventional product implemented |
| §6.2 Pilot | Full journey + real borrowers | All 9 stages + compliance |
| §6.3 Ramp | All products + nationwide | FHA/VA/Jumbo/ARM in eligibility engine |

---

## Quick Start

### 1. Clone and setup

```bash
cd mortgage_ai
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY and LANGCHAIN_API_KEY
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Ingest knowledge base

```bash
python scripts/ingest_knowledge.py
# Output: "Done! Ingested 127 document chunks."
```

### 4. Start the API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Copy .env.example → .env with your API keys
docker compose up --build

# API available at: http://localhost:8000
# ChromaDB at:      http://localhost:8001
# Swagger UI at:    http://localhost:8000/docs
```

---

## API Walkthrough — Happy Path

### 1. Create Application
```bash
curl -X POST http://localhost:8000/api/v1/applications \
  -H "Content-Type: application/json" \
  -d '{
    "borrower_name": "Alex Johnson",
    "borrower_email": "alex@example.com",
    "loan_purpose": "purchase"
  }'
# Returns: application_id, welcome_message
```

### 2. Run Pre-Qualification (≤ 60 seconds, NFR-002)
```bash
curl -X POST http://localhost:8000/api/v1/applications/{id}/prequal \
  -H "Content-Type: application/json" \
  -d '{
    "annual_income": 120000,
    "monthly_debt_payments": 500,
    "estimated_home_price": 450000,
    "down_payment": 90000,
    "credit_score_range_min": 720,
    "loan_purpose": "purchase"
  }'
# Returns: eligible, max_loan, rate_range, prequal_letter_url
```

### 3. Conversational Chat
```bash
curl -X POST http://localhost:8000/api/v1/applications/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What documents do I need to upload?"}'
# Returns: AI response with document requirements
```

### 4. Streaming Chat (SSE)
```bash
curl "http://localhost:8000/api/v1/applications/{id}/stream?message=explain+my+rate+options"
# Server-Sent Events stream: token-by-token response + stage change events
```

### 5. Upload Document
```bash
curl -X POST http://localhost:8000/api/v1/applications/{id}/documents \
  -F "filename=paystub.pdf" \
  -F "file_size_kb=150" \
  -F "raw_text_excerpt=Pay Period May 2026 Gross Pay 5000 Net Pay 3500 YTD 45000 Acme Corp" \
  -F "full_text=..."
# Returns: doc_type, verified, extracted_data, feedback_message
```

### 6. Knowledge Base Search
```bash
curl "http://localhost:8000/api/v1/knowledge/search?q=what+is+DTI&n=3"
# Returns: semantic search results from mortgage knowledge base
```

### 7. Loan Officer Copilot
```bash
curl -X POST http://localhost:8000/api/v1/lo/analyze \
  -H "Content-Type: application/json" \
  -d '{"application_id": "{id}", "lo_question": "Are there any compliance concerns?"}'
# Returns: AI file analysis, health score, compliance alerts
```

---

## LangSmith Observability

Every LLM call, tool invocation, and graph node execution is automatically traced in LangSmith:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=b2c-mortgage-ai
```

View traces at: `https://smith.langchain.com`

Tracked metrics per BRD §5.8:
- Pre-qualification response time (target: ≤ 60s)
- Tool call success/failure rates
- LLM token usage per stage
- Agent ReAct iteration counts
- Stage completion rates (abandonment funnel)

---

## Project Structure

```
mortgage_ai/
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic settings (env vars)
│   ├── models/
│   │   └── loan.py              # Domain models (LoanApplication, etc.)
│   ├── tools/                   # LangChain @tool functions
│   │   ├── credit_tools.py      # Soft/hard credit check
│   │   ├── eligibility_tools.py # DTI/LTV/FICO rules engine
│   │   ├── document_tools.py    # OCR classify/extract/validate
│   │   ├── rate_tools.py        # Payment/APR/rate-lock
│   │   ├── compliance_tools.py  # TRID/HMDA/ECOA/E-SIGN
│   │   └── knowledge_tools.py   # RAG knowledge search
│   ├── vectorstore/
│   │   └── knowledge_base.py    # ChromaDB + embeddings
│   ├── graphs/
│   │   ├── states.py            # MortgageState TypedDict + defaults
│   │   └── mortgage_graph.py    # LangGraph 9-stage workflow
│   ├── agents/
│   │   ├── prequalification_agent.py  # Stage 2 specialist
│   │   ├── document_agent.py          # Stage 4 specialist
│   │   └── loan_officer_agent.py      # LO copilot
│   ├── skills/
│   │   ├── mortgage_advisor_skill.py  # LCEL RAG chain
│   │   ├── prequalification_skill.py  # Composable prequal pipeline
│   │   └── document_extraction_skill.py  # Bulk doc pipeline
│   └── api/
│       ├── main.py              # FastAPI app factory
│       └── routes.py            # All API endpoints
├── data/
│   └── knowledge/
│       ├── mortgage_products.md # Loan product descriptions
│       ├── regulations.md       # TRID/ECOA/HMDA/FCRA/GLBA
│       └── faq.md              # Common borrower questions
├── tests/
│   ├── test_tools.py            # Tool unit tests (no API calls)
│   ├── test_skills.py           # Skill pipeline tests (mocked LLM)
│   └── test_graph.py            # LangGraph state/routing tests
├── scripts/
│   └── ingest_knowledge.py      # One-time knowledge base ingestion
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── .env.example
```

---

## Key Design Decisions

### LangGraph vs. simple chains
The 9-stage mortgage workflow has genuine state machine behavior — conditional routing, loops (document re-requests), human-in-the-loop checkpoints, and cross-device resume. LangGraph's `StateGraph` + `MemorySaver` maps directly to these requirements.

### ChromaDB over managed vector DB
ChromaDB runs locally with no external service for development/POC (BRD §6.1). In production (Pilot/Ramp-Up), swap to `PGVector` (PostgreSQL) or `Pinecone` by changing `knowledge_base.py`.

### Anthropic Claude as primary LLM
Claude Sonnet 4.6 for complex reasoning (underwriting, compliance, document analysis); Claude Haiku 4.5 for fast FAQ/status responses. Both accessed via `langchain-anthropic` with automatic LangSmith tracing.

### Sandbox tool stubs
All third-party integrations (credit bureaus, AUS, appraisal) are simulated sandbox stubs that return realistic data. Replace `_call_bureau()` functions with real API clients for production.

---

## Regulatory Compliance

The platform implements mandatory requirements from BRD §5.4:

| Regulation | Implementation |
|---|---|
| TRID (Reg Z + X) | `check_trid_compliance` — enforces 3-day LE and 3-day CD timing |
| ECOA (Reg B) | `check_ecoa_compliance` — 30-day adverse action notice |
| HMDA (Reg C) | `check_hmda_data` — demographic data + opt-out |
| FCRA | Consent gates on hard pull + credit score disclosure |
| GLBA | Data encryption at rest (AES-256) + in transit (TLS 1.3) |
| E-SIGN / UETA | `capture_esign_consent` — required before any electronic disclosure |

**Disclosure timing is non-negotiable** (BRD §7.2) — the compliance tools always run before disclosure delivery.

---

## Delivery Phases

| Phase | Status | Scope |
|---|---|---|
| POC | Ready | Conventional 30yr, web, happy path, 50-100 testers |
| Pilot | Foundation ready | All products, 5-8 states, real borrowers |
| Ramp-Up | Architecture supports | Nationwide, iOS/Android, RON |

---

## Contributing

See the root repository `CODE_OF_CONDUCT.md` and `CHANGELOG.md` for contribution guidelines.

```bash
# Run tests before submitting
pytest tests/ -v

# Check types (optional)
mypy src/ --ignore-missing-imports
```
