"""Mortgage Advisor Skill — LCEL RAG chain for contextual borrower help (FR-054).

A "skill" here is a composable, reusable LCEL chain that can be called
from any agent node in the LangGraph or directly from the API.

This skill:
  1. Takes a borrower question + optional application context
  2. Retrieves relevant chunks from the ChromaDB knowledge base
  3. Constructs a context-aware prompt
  4. Returns a plain-language answer with source attribution

LangSmith traces the full chain automatically.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.config import settings
from src.vectorstore.knowledge_base import get_knowledge_base


ADVISOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly, knowledgeable mortgage advisor helping a borrower understand
the mortgage process. You answer questions using the provided mortgage knowledge context.

Rules:
- Plain language (8th-grade reading level)
- Cite the source regulation or product name when relevant
- If the answer is not in the context, say so and offer to connect them with a loan officer
- Never give specific financial advice — provide general education
- Keep answers under 150 words unless more detail is explicitly requested
- Be encouraging and empathetic

Mortgage Knowledge Context:
{context}

Application Context (if any):
{application_context}""",
    ),
    ("human", "{question}"),
])


def _build_llm():
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.fast_llm_model,  # Use fast model for FAQ answers
            api_key=settings.anthropic_api_key,
            temperature=0.1,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


class MortgageAdvisorSkill:
    """RAG-powered mortgage advisor skill for contextual borrower help."""

    def __init__(self):
        self._llm = _build_llm()
        self._kb = get_knowledge_base()
        self._chain = self._build_chain()

    def _build_chain(self):
        """Build the LCEL RAG chain."""
        retriever = self._kb.as_retriever()

        def format_docs(docs) -> str:
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": RunnablePassthrough() | (lambda x: x["question"]),
                "application_context": RunnablePassthrough() | (
                    lambda x: x.get("application_context", "No application context provided.")
                ),
            }
            | ADVISOR_PROMPT
            | self._llm
            | StrOutputParser()
        )
        return chain

    def answer(self, question: str, application_context: dict | None = None) -> str:
        """Answer a borrower question using RAG.

        Args:
            question: The borrower's question in natural language
            application_context: Optional dict with loan_amount, stage, etc.

        Returns:
            Plain-language answer string
        """
        context_str = ""
        if application_context:
            parts = []
            if "stage" in application_context:
                parts.append(f"Current stage: {application_context['stage']}")
            if "loan_amount" in application_context:
                parts.append(f"Loan amount: ${application_context['loan_amount']:,.0f}")
            if "selected_product" in application_context:
                parts.append(f"Selected product: {application_context['selected_product']}")
            context_str = " | ".join(parts)

        try:
            return self._chain.invoke({
                "question": question,
                "application_context": context_str or "General inquiry (no active application)",
            })
        except Exception as e:
            return (
                f"I'm sorry, I had trouble retrieving information for your question. "
                f"Please try again or contact your loan officer for assistance."
            )

    def stream_answer(self, question: str, application_context: dict | None = None):
        """Stream the answer token by token (for SSE endpoints)."""
        context_str = ""
        if application_context:
            context_str = f"Stage: {application_context.get('stage', 'unknown')}"

        yield from self._chain.stream({
            "question": question,
            "application_context": context_str,
        })

    def get_faq_answers(self, topics: list[str]) -> dict[str, str]:
        """Bulk answer common FAQ topics for pre-population."""
        return {topic: self.answer(topic) for topic in topics}
