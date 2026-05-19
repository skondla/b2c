"""One-time script to ingest mortgage knowledge base into ChromaDB.

Run from the mortgage_ai/ directory:
    python scripts/ingest_knowledge.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore.knowledge_base import initialize_knowledge_base

if __name__ == "__main__":
    print("Ingesting mortgage knowledge base into ChromaDB...")
    count = initialize_knowledge_base("data/knowledge")
    print(f"Done! Ingested {count} document chunks.")
    print("Knowledge base is ready for semantic search.")
