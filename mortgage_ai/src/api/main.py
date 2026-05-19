"""FastAPI application entry point for the B2C Mortgage AI Platform.

Exposes RESTful endpoints for all 9 journey stages plus:
  • SSE streaming for real-time AI responses
  • Loan Officer copilot API
  • Knowledge base search
  • Document upload and processing
  • Application state management

NFR compliance:
  - Page load / API response ≤ 500 ms p95 (NFR-005)
  - ≥ 10,000 concurrent applicants via async (NFR-004)
  - TLS in production, security headers (NFR-020)
  - Structured logging with correlation IDs (§5.8)

LangSmith tracing: enabled via LANGCHAIN_TRACING_V2=true in environment.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Set LangSmith env vars from settings
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    logger.info(
        "B2C Mortgage AI Platform starting",
        env=settings.app_env,
        langsmith_project=settings.langchain_project,
        tracing=settings.langchain_tracing_v2,
    )

    # Pre-warm knowledge base (ingest markdown files if not done)
    try:
        from src.vectorstore.knowledge_base import get_knowledge_base, initialize_knowledge_base
        kb = get_knowledge_base()
        count = initialize_knowledge_base("data/knowledge")
        if count > 0:
            logger.info("Knowledge base ingested", chunks=count)
        else:
            logger.info("Knowledge base already populated")
    except Exception as e:
        logger.warning("Knowledge base initialization skipped", error=str(e))

    yield

    logger.info("B2C Mortgage AI Platform shutting down")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="B2C Mortgage Loan Application AI Platform",
        description=(
            "AI-powered mortgage application platform with LangChain, LangGraph, "
            "LangSmith, and ChromaDB vector search. Supports 9-stage borrower journey "
            "with LLM-powered guidance, document verification, and compliance automation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (restrict in production) ─────────────────────────────────────────
    origins = ["*"] if not settings.is_production else [
        "https://mortgage.example.com",
        "https://app.mortgage.example.com",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Correlation ID middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        start = time.monotonic()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        # NFR-005: log if p95 target exceeded
        if elapsed_ms > settings.api_response_p95_ms:
            logger.warning("Slow response", elapsed_ms=elapsed_ms, path=request.url.path)

        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        correlation_id = request.headers.get("X-Correlation-ID", "unknown")
        logger.error("Unhandled exception", error=str(exc), correlation_id=correlation_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "correlation_id": correlation_id,
            },
        )

    # ── Include routers ───────────────────────────────────────────────────────
    from src.api.routes import router
    app.include_router(router, prefix="/api/v1")

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        return {
            "status": "healthy",
            "env": settings.app_env,
            "version": "1.0.0",
            "langsmith_project": settings.langchain_project,
        }

    return app


app = create_app()
