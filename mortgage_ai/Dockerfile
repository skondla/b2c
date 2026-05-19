FROM python:3.13-slim

WORKDIR /app

# System dependencies for sentence-transformers and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/chroma_db data/knowledge

# Ingest knowledge base at build time
RUN python scripts/ingest_knowledge.py || echo "Knowledge ingestion will run at startup"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
