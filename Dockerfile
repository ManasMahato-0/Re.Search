FROM python:3.12-slim

# HF Spaces runs the container as user 1000 — create it and own /app
RUN useradd -m -u 1000 user
WORKDIR /app

# Deploy defaults: CPU + lighter reranker + smaller pool for free-tier hardware
ENV RERANKER_MODEL=BAAI/bge-reranker-base \
    DEVICE=cpu \
    RERANK_POOL=15 \
    HF_HOME=/app/hf-cache

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so cold boots skip the ~500MB fetch
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('BAAI/bge-reranker-base')"

COPY backend/ .
RUN chown -R user:user /app

USER user

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
