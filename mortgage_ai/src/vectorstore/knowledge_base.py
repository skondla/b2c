"""ChromaDB-backed mortgage knowledge base with semantic vector search.

Two collections:
  1. mortgage_knowledge  — regulations, products, FAQ, glossary (RAG for agents)
  2. borrower_documents  — extracted borrower doc content (per-application retrieval)

Embeddings: HuggingFace sentence-transformers (local, no API key required).
Falls back to OpenAI embeddings if OPENAI_API_KEY is set and model is configured.

LangSmith tracing is enabled automatically via LANGCHAIN_TRACING_V2=true.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def _build_embeddings():
    """Return embedding model — HuggingFace local by default."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        pass

    # Fallback: OpenAI embeddings
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.openai_api_key)

    raise RuntimeError(
        "No embedding provider available. Install langchain-huggingface or set OPENAI_API_KEY."
    )


class MortgageKnowledgeBase:
    """Singleton knowledge base wrapping ChromaDB + LangChain retrieval."""

    def __init__(self):
        self._embeddings = _build_embeddings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._knowledge_store: Optional[Chroma] = None
        self._doc_store: Optional[Chroma] = None

    def _get_knowledge_store(self) -> Chroma:
        if self._knowledge_store is None:
            self._knowledge_store = Chroma(
                client=self._client,
                collection_name=settings.knowledge_collection_name,
                embedding_function=self._embeddings,
            )
        return self._knowledge_store

    def _get_doc_store(self) -> Chroma:
        if self._doc_store is None:
            self._doc_store = Chroma(
                client=self._client,
                collection_name=settings.doc_collection_name,
                embedding_function=self._embeddings,
            )
        return self._doc_store

    def ingest_knowledge_files(self, knowledge_dir: str = "data/knowledge") -> int:
        """Ingest all Markdown files from the knowledge directory into ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )

        docs: list[Document] = []
        md_files = glob.glob(os.path.join(knowledge_dir, "**/*.md"), recursive=True)

        for filepath in md_files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Infer category from filename
            fname = Path(filepath).stem
            if "product" in fname:
                category = "products"
            elif "regulat" in fname or "compliance" in fname:
                category = "regulations"
            elif "faq" in fname:
                category = "faq"
            elif "glossar" in fname:
                category = "glossary"
            else:
                category = "general"

            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": filepath,
                        "category": category,
                        "chunk_index": i,
                        "filename": fname,
                    },
                ))

        if not docs:
            return 0

        store = self._get_knowledge_store()
        store.add_documents(docs)
        return len(docs)

    def search(
        self,
        query: str,
        n_results: int = 3,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Semantic similarity search over the mortgage knowledge base."""
        store = self._get_knowledge_store()
        where = {"category": category} if category else None

        try:
            results = store.similarity_search_with_relevance_scores(
                query,
                k=n_results,
                filter=where,
            )
        except Exception:
            # Collection may be empty — return empty
            return []

        return [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "category": doc.metadata.get("category", "general"),
                "source": doc.metadata.get("filename", "unknown"),
            }
            for doc, score in results
        ]

    def add_borrower_document(
        self,
        application_id: str,
        document_id: str,
        text: str,
        doc_type: str,
    ) -> None:
        """Store extracted borrower document text for per-application retrieval."""
        store = self._get_doc_store()
        store.add_documents([
            Document(
                page_content=text,
                metadata={
                    "application_id": application_id,
                    "document_id": document_id,
                    "doc_type": doc_type,
                },
            )
        ])

    def search_borrower_docs(
        self,
        application_id: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """Retrieve borrower documents relevant to a query, scoped to application."""
        store = self._get_doc_store()
        try:
            results = store.similarity_search_with_relevance_scores(
                query,
                k=n_results,
                filter={"application_id": application_id},
            )
        except Exception:
            return []

        return [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "doc_type": doc.metadata.get("doc_type"),
                "document_id": doc.metadata.get("document_id"),
            }
            for doc, score in results
        ]

    def as_retriever(self, category: Optional[str] = None):
        """Return a LangChain retriever for use in LCEL chains."""
        store = self._get_knowledge_store()
        search_kwargs: dict = {"k": 4}
        if category:
            search_kwargs["filter"] = {"category": category}
        return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


# ── Singleton ─────────────────────────────────────────────────────────────────

_kb_instance: Optional[MortgageKnowledgeBase] = None


def get_knowledge_base() -> MortgageKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = MortgageKnowledgeBase()
    return _kb_instance


def initialize_knowledge_base(knowledge_dir: str = "data/knowledge") -> int:
    """One-time ingestion call — run from scripts/ingest_knowledge.py."""
    kb = get_knowledge_base()
    count = kb.ingest_knowledge_files(knowledge_dir)
    return count
