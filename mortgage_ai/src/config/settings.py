from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM Providers ─────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Primary LLM model (Claude Sonnet 4.6 for complex reasoning)
    primary_llm_model: str = "claude-sonnet-4-6"
    fast_llm_model: str = "claude-haiku-4-5-20251001"  # Fast/cheap tasks

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="b2c-mortgage-ai", alias="LANGCHAIN_PROJECT")

    # ── Vector Store ──────────────────────────────────────────────────────────
    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_collection_name: str = "mortgage_knowledge"
    doc_collection_name: str = "borrower_documents"

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    log_level: str = "INFO"

    # ── Third-Party Stubs ─────────────────────────────────────────────────────
    credit_bureau_api_key: str = "sandbox-key"
    credit_bureau_url: str = "https://sandbox.creditbureau.example.com"
    aus_api_key: str = "sandbox-key"
    aus_url: str = "https://sandbox.aus.example.com"
    appraisal_api_key: str = "sandbox-key"
    plaid_client_id: str = "sandbox"
    plaid_secret: str = "sandbox"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/mortgage.db"

    # ── Performance / NFR targets (NFR-001 through NFR-005) ───────────────────
    max_concurrent_applicants: int = 10_000
    prequal_timeout_seconds: int = 60        # NFR-002: ≤ 60 s
    doc_upload_max_mb: int = 25             # NFR-003: per-file limit
    api_response_p95_ms: int = 500          # NFR-005: ≤ 500 ms

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()
