"""Knowledge base search tool — RAG over mortgage regulations, products, and FAQs.

Uses the ChromaDB vector store initialized by src/vectorstore/knowledge_base.py.
Provides contextual help throughout the application (FR-054, BRD §3.9).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="Borrower's question or topic to search in the mortgage knowledge base")
    n_results: int = Field(default=3, description="Number of knowledge chunks to retrieve (1-5)")
    filter_category: str = Field(
        default="",
        description="Optional filter: products, regulations, faq, glossary, or empty for all"
    )


class ProductInfoInput(BaseModel):
    product_name: str = Field(
        description="Loan product name: conventional_30_fixed, fha_30_fixed, va_30_fixed, jumbo_30_fixed, etc."
    )


@tool("search_mortgage_knowledge", args_schema=KnowledgeSearchInput, return_direct=False)
def search_mortgage_knowledge(query: str, n_results: int = 3, filter_category: str = "") -> dict:
    """Search the mortgage knowledge base using semantic vector search (RAG).
    Answers borrower questions about mortgage products, regulations, process steps,
    and requirements. Used throughout the application for contextual help (FR-054).
    """
    try:
        from src.vectorstore.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        results = kb.search(query, n_results=min(n_results, 5), category=filter_category or None)
        return {
            "query": query,
            "results": results,
            "source": "mortgage_knowledge_base",
        }
    except Exception as e:
        # Fallback: return static answer if vector store not initialized
        return {
            "query": query,
            "results": [_fallback_answer(query)],
            "source": "fallback_static",
            "note": f"Vector store unavailable ({e}); returning static answer.",
        }


@tool("get_product_info", args_schema=ProductInfoInput, return_direct=False)
def get_product_info(product_name: str) -> dict:
    """Retrieve detailed information about a specific mortgage loan product
    including eligibility criteria, rates, terms, and key features.
    """
    from .eligibility_tools import PRODUCT_RULES

    product_descriptions = {
        "conventional_30_fixed": {
            "display_name": "30-Year Fixed Conventional",
            "description": (
                "Most popular mortgage type. Fixed rate for the full 30-year term means "
                "predictable payments. Best for buyers planning to stay 7+ years."
            ),
            "pros": [
                "Lowest monthly payment of fixed products",
                "Predictable payment — rate never changes",
                "PMI removable once LTV < 80%",
                "No upfront mortgage insurance",
            ],
            "cons": [
                "Higher total interest paid vs. 15-year",
                "PMI required if down payment < 20%",
            ],
            "ideal_for": "First-time buyers, long-term homeowners",
        },
        "conventional_15_fixed": {
            "display_name": "15-Year Fixed Conventional",
            "description": "Pay off your home in half the time at a lower rate than the 30-year.",
            "pros": [
                "Lower interest rate (~0.5-0.75% less than 30-year)",
                "Build equity faster",
                "Significantly less total interest paid",
            ],
            "cons": [
                "Higher monthly payment",
                "Less cash flow flexibility",
            ],
            "ideal_for": "Refinancers with established income, buyers with large down payments",
        },
        "fha_30_fixed": {
            "display_name": "FHA 30-Year Fixed",
            "description": (
                "Government-backed loan requiring only 3.5% down. "
                "Accepts lower credit scores than conventional loans."
            ),
            "pros": [
                "Down payment as low as 3.5%",
                "Accepts FICO scores from 500 (with 10% down) or 580 (with 3.5% down)",
                "More flexible debt-to-income limits",
            ],
            "cons": [
                "Upfront MIP of 1.75% added to loan",
                "Annual MIP for life of loan (if <10% down)",
                "Loan limits apply by county",
            ],
            "ideal_for": "First-time buyers with lower credit scores or limited savings",
        },
        "va_30_fixed": {
            "display_name": "VA 30-Year Fixed",
            "description": (
                "Exclusively for eligible veterans, active duty, and surviving spouses. "
                "No down payment or PMI required."
            ),
            "pros": [
                "Zero down payment required",
                "No private mortgage insurance",
                "Competitive rates",
                "Easier qualification guidelines",
            ],
            "cons": [
                "VA funding fee (2.3% for first use, can be financed)",
                "Must be VA-eligible borrower",
                "Property must meet VA minimum property requirements",
            ],
            "ideal_for": "Eligible veterans and military families",
        },
        "jumbo_30_fixed": {
            "display_name": "Jumbo 30-Year Fixed",
            "description": "For loan amounts exceeding conforming loan limits ($766,550 in most markets).",
            "pros": [
                "Finances high-value properties",
                "Competitive rates for well-qualified borrowers",
            ],
            "cons": [
                "Stricter qualification (FICO 700+, 20% down)",
                "12 months reserves typically required",
                "Larger down payment required",
            ],
            "ideal_for": "High-income borrowers in high-cost markets",
        },
    }

    rules = PRODUCT_RULES.get(product_name, {})
    info = product_descriptions.get(product_name, {"description": "Product information not found"})

    return {
        "product": product_name,
        **info,
        "eligibility_requirements": {
            "minimum_fico": rules.get("min_fico"),
            "maximum_dti": f"{rules.get('max_dti', 0):.0%}",
            "maximum_ltv": f"{rules.get('max_ltv', 0):.0%}",
            "maximum_loan_amount": f"${rules.get('max_loan', 0):,.0f}",
        },
        "current_rate_estimate": f"{rules.get('rate_base', 0):.3f}%",
    }


def _fallback_answer(query: str) -> dict:
    """Static fallback answers for common mortgage questions."""
    q = query.lower()
    if "dti" in q or "debt" in q:
        return {
            "content": (
                "Debt-to-Income (DTI) ratio is your total monthly debt payments divided by your gross monthly income. "
                "Most conventional loans allow up to 45% DTI. FHA allows up to 57%. "
                "Lower DTI improves your chances of approval and gets you better rates."
            ),
            "category": "faq",
        }
    if "down payment" in q:
        return {
            "content": (
                "Down payment requirements: Conventional loans need as little as 3% (with PMI). "
                "FHA requires 3.5% (580+ FICO) or 10% (500-579 FICO). "
                "VA loans require 0% down for eligible veterans. "
                "Putting 20% down eliminates PMI on conventional loans."
            ),
            "category": "faq",
        }
    if "close" in q or "closing" in q:
        return {
            "content": (
                "Closing costs typically run 2-5% of the loan amount. "
                "They include: origination fee, appraisal, title insurance, recording fees, and prepaid items. "
                "You'll receive a Loan Estimate within 3 business days of application, "
                "and a Closing Disclosure at least 3 business days before closing."
            ),
            "category": "faq",
        }
    return {
        "content": (
            "For specific mortgage questions, our loan officers are available to help. "
            "You can schedule a call through your application dashboard or message us through the secure portal."
        ),
        "category": "general",
    }
