# Hybrid Search Engine v1.0

A modern hybrid search engine that combines traditional Information Retrieval techniques with semantic vector search to provide more relevant search results.

The engine performs retrieval using **BM25** and **dense embeddings**, merges both candidate sets using **Reciprocal Rank Fusion (RRF)**, and reranks the final results using a **Cross-Encoder** to improve ranking quality.

This project was built to explore modern search engine architecture from scratch while understanding the reasoning behind each retrieval component rather than relying solely on existing search frameworks.

---

# Features

### Lexical Retrieval

- BM25 Ranking
- Custom tokenizer
- Stopword removal
- Title-aware indexing

### Semantic Retrieval

- BAAI/bge-small-en-v1.5 embeddings
- FAISS vector index
- Cosine similarity search

### Hybrid Retrieval

- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Query-aware snippet generation

### Backend

- FastAPI REST API

### Frontend

- React
- TailwindCSS

---

# Retrieval Pipeline

```
                    Documents
                         │
                Document Processing
                         │
            Chunking + Tokenization
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   BM25 Lexical Index              Dense Embeddings
        │                                 │
        │                           FAISS Index
        └───────────────┬─────────────────┘
                        │
             Candidate Retrieval
                        │
        Reciprocal Rank Fusion (RRF)
                        │
          Cross-Encoder Reranking
                        │
          Query-aware Snippets
                        │
                 Search Results
```

---

# Evolution

## Version 0.1

Initial proof of concept.

Implemented

- BM25 lexical retrieval
- SentenceTransformer embeddings
- FAISS vector search
- Basic hybrid score fusion
- FastAPI backend

Although functional, retrieval quality was inconsistent and several limitations became apparent.

---

## Version 1.0

Major retrieval improvements.

Added

- BAAI BGE retrieval embeddings
- Cosine similarity search
- Chunk-based indexing
- Overlapping chunks
- Title-aware indexing
- Shared tokenizer
- URL deduplication
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Query-aware snippet generation
- Modular code structure

These improvements significantly improved retrieval quality while keeping the system lightweight enough to run on consumer hardware.

---

# Tech Stack

## Backend

- Python
- FastAPI

## Retrieval

- BM25
- FAISS
- SentenceTransformers
- BAAI/bge-small-en-v1.5
- CrossEncoder (MS MARCO MiniLM)

## Frontend

- React
- TailwindCSS

## Crawling

- Scrapy

---

# Project Structure

```
Hybrid-Search-Engine/

│
├── backend/
│   │
│   ├── data/
│   ├── bm25.py
│   ├── indexer.py
│   ├── main.py
│   ├── requirements.txt
│   ├── vector_index.faiss   (generated locally — not in repo)
│   └── bm25_index.pkl       (generated locally — not in repo)
│
├── crawler/
│   │
│   ├── crawler/
│   ├── docs_ingester.py
│   ├── docs_data.json
│   ├── papers_data.json
│   └── scrapy.cfg
│
├── frontend/
│
├── README.md
│
└── .gitignore
```

---

# Getting Started

Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git

cd <repo-name>
```

Install backend dependencies

```bash
cd backend

pip install -r requirements.txt
```

Install frontend dependencies

```bash
cd ../frontend

npm install
```

---

# Running

> **Note:** Build the search indexes first — see [Dataset & Indexes](#dataset--indexes).

Backend

```bash
cd backend

uvicorn main:app --reload
```

Frontend

```bash
cd frontend

npm run dev
```

---

# Dataset & Indexes

## What is excluded from the repository

The following files exceed GitHub's file size limits and are **not** included in this repository (see `.gitignore`):

| File | Size | Purpose |
|------|------|---------|
| `backend/bm25_index.pkl` | ~271 MB | BM25 lexical index |
| `backend/vector_index.faiss` | ~131 MB | FAISS dense vector index |
| `backend/data/crawler_dataset.jsonl` | ~72 MB | Crawled documents dataset |

The repository **does** include `backend/data/research_dataset.json` (~22 MB), so the engine can be bootstrapped from it alone.

## How to build the indexes before running

The backend will not start without the indexes. After installing dependencies, generate them:

1. (Optional) Build a larger dataset — fetch documents from arXiv, OpenAlex, Wikipedia, StackExchange, W3C/RFC:

   ```bash
   cd crawler
   python bulk_ingester.py
   ```

   You can also run the Scrapy spiders to produce `backend/data/crawler_dataset.jsonl`. If this file is missing, the indexer simply skips it.

2. Build the BM25 and FAISS indexes (required):

   ```bash
   cd backend
   python indexer.py
   ```

   This reads the datasets in `backend/data/` and produces `bm25_index.pkl` and `vector_index.faiss`. The first run also downloads the embedding model (BAAI/bge-small-en-v1.5).

3. Start the backend as described in the [Running](#running) section.

Re-run `python indexer.py` whenever documents are added or the retrieval pipeline changes.

---

# Current Capabilities

- Hybrid lexical + semantic retrieval
- Dense vector search
- Candidate fusion
- Cross-encoder reranking
- Query-aware snippets
- Chunk-based indexing
- REST API

---

# Future Improvements

This project is still actively under development.

Future versions aim to improve retrieval quality, scalability and search efficiency through more advanced indexing techniques and modern Information Retrieval algorithms.

---

# Screenshots

_Coming soon_

---

# License

MIT License