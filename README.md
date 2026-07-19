# Hybrid Search Engine v2.0

A hybrid search engine combining lexical and semantic retrieval: **BM25** over an inverted index, **dense embeddings** via FAISS, **weighted Reciprocal Rank Fusion**, and **cross-encoder reranking** — with a measurement-driven eval harness that guided every architecture decision.

Built from scratch to understand modern search architecture component by component, rather than relying on existing search frameworks.

**Live demo:** _[add your Space / Vercel URLs here]_

---

# Features

### Lexical Retrieval

- Okapi BM25 with inverted-index (postings) scoring — only chunks containing query terms are scored
- Custom tokenizer with stopword removal
- Title-aware indexing

### Semantic Retrieval

- BAAI/bge-small-en-v1.5 embeddings
- FAISS vector index, cosine similarity
- Similarity-cutoff noise filter

### Hybrid Ranking

- Weighted Reciprocal Rank Fusion (RRF) across engines and query variants
- Title-aware cross-encoder reranking (BAAI/bge-reranker-v2-m3, fp16 on GPU)
- Query-aware snippet generation

### Query Expansion (experimental, off by default)

- Local LLM (Qwen2.5-1.5B-Instruct) rewrites queries: acronym expansion, typo fixes, concept naming
- Multi-query retrieval fused with weighted RRF; original query always anchors
- Disk-cached expansions
- Measured on the eval sets: bridges retrieval vocab gaps but gains were erased at the rerank stage — disabled by default (`QUERY_EXPANSION=1` re-enables)

### Evaluation Harness

- nDCG@10, MRR@10, recall@pool, latency percentiles
- Easy set (63 queries) + adversarial hard set (20 queries: typos, acronyms, indirect concept queries)
- Per-query failure buckets: retrieval misses vs ranking misses
- Tagged, persisted runs for regression tracking

---

# Retrieval Pipeline

```
                     Query
                       │
        ┌── (optional) LLM query expansion ──┐
        │                                    │
        ▼                                    ▼
   BM25 inverted index              FAISS cosine search
        │                                    │
        └───────────────┬────────────────────┘
                        │
          Weighted Reciprocal Rank Fusion
                        │
              Chunk → document collapse
                        │
       Cross-encoder rerank (title + chunk)
                        │
             Query-aware snippets
                        │
                 Search results
```

---

# Eval Results

| metric | easy (63 q) | hard (20 q) |
|--------|------------|-------------|
| nDCG@10 | 0.95 | 0.69 |
| MRR@10 | 0.93 | 0.65 |
| recall@30 | 1.00 | 0.85 |

Hard-set queries include deliberate typos ("kubernets", "distrbuted"), bare acronyms ("rmsle metric"), and indirect concept descriptions ("why robots bad at easy things").

---

# Evolution

## Version 0.1

Proof of concept: BM25 + SentenceTransformer embeddings + basic score fusion, FastAPI backend.

## Version 1.0

Retrieval quality: BGE embeddings, chunk-based indexing with overlap, title-aware indexing, RRF fusion, cross-encoder reranking, query-aware snippets.

## Version 2.0

Measurement-driven improvements:

- **Eval harness** with easy + adversarial hard query sets — every change below was accepted or rejected on measured nDCG/MRR/recall deltas
- **Title-aware reranking** — cross-encoder scores `title + chunk` pairs; page identity was previously invisible to the reranker
- **Reranker upgrade** to bge-reranker-v2-m3 (fp16, GPU) — the single largest quality win (hard nDCG 0.61 → 0.69)
- **Inverted-index BM25** — postings-based sparse scoring replaces full-corpus scans
- **LLM query expansion** — built, measured, and honestly parked: retrieval recall improved but final ranking didn't, so it ships disabled behind an env flag
- **Deploy support** — Dockerfile for Hugging Face Spaces (CPU deploy profile via env knobs), configurable frontend API URL

---

# Tech Stack

**Backend:** Python, FastAPI, Uvicorn
**Retrieval:** BM25 (custom), FAISS, SentenceTransformers, BAAI/bge-small-en-v1.5, BAAI/bge-reranker-v2-m3
**Query expansion:** Transformers, Qwen2.5-1.5B-Instruct (optional)
**Frontend:** React, TailwindCSS
**Crawling:** Scrapy
**Eval:** custom harness (nDCG / MRR / recall / latency)

---

# Project Structure

```
Search-Engine/
│
├── backend/
│   ├── data/
│   ├── bm25.py               # BM25 + inverted index
│   ├── indexer.py            # builds bm25_index.pkl + vector_index.faiss
│   ├── main.py               # FastAPI app + retrieval pipeline
│   ├── query_expansion.py    # optional LLM multi-query expansion
│   ├── requirements.txt
│   ├── vector_index.faiss    (generated locally — not in repo)
│   └── bm25_index.pkl        (generated locally — not in repo)
│
├── crawler/                  # Scrapy spiders + bulk ingester
├── eval/
│   ├── run_eval.py           # eval harness
│   ├── queries.jsonl         # easy set
│   ├── queries_hard.jsonl    # adversarial set
│   └── results/              # tagged runs
│
├── frontend/                 # React + Tailwind UI
├── Dockerfile                # HF Spaces CPU deploy
└── README.md
```

---

# Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# backend deps
cd backend
pip install -r requirements.txt

# frontend deps
cd ../frontend
npm install
```

---

# Building the Indexes

The backend will not start without the indexes (too large for the repo — see `.gitignore`).

1. (Optional) Build a larger dataset — arXiv, OpenAlex, Wikipedia, StackExchange, W3C/RFC:

   ```bash
   cd crawler
   python bulk_ingester.py
   ```

2. Build the BM25 and FAISS indexes (required):

   ```bash
   cd backend
   python indexer.py
   ```

   Reads datasets in `backend/data/`, produces `bm25_index.pkl` and `vector_index.faiss`. First run downloads the embedding model.

Re-run `indexer.py` whenever documents are added.

---

# Running

Backend:

```bash
cd backend
uvicorn main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Point the frontend at a non-local backend with `VITE_API_URL`.

---

# Evaluation

```bash
cd backend
python ../eval/run_eval.py                                       # easy set
python ../eval/run_eval.py --queries ../eval/queries_hard.jsonl  # hard set
python ../eval/run_eval.py --tag my-experiment                   # tagged run
```

Results persist to `eval/results/` as JSON with per-query breakdowns.

---

# Configuration

Environment knobs (defaults target local GPU dev):

| var | default | purpose |
|-----|---------|---------|
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | cross-encoder model |
| `DEVICE` | cuda | reranker device |
| `RERANK_POOL` | 30 | docs rescored by cross-encoder |
| `QUERY_EXPANSION` | 0 | enable LLM multi-query expansion |
| `MIN_COSINE_SIM` | 0.4 | FAISS similarity cutoff |

The Dockerfile ships a CPU profile (bge-reranker-base, pool 15) for free-tier hosting.

---

# Deployment (free tier)

**Backend — Hugging Face Spaces (Docker SDK):**

1. Create a Space, Docker SDK, CPU basic (free)
2. Copy `Dockerfile`, `backend/*.py`, `backend/requirements.txt`, and both index files into the Space repo
3. `git lfs track "*.pkl" "*.faiss"` before committing the indexes
4. Push — the image build pre-downloads models, so cold boots are fast
5. API serves at `https://<user>-<space>.hf.space/search?q=...`

**Frontend — Vercel:**

1. Import the GitHub repo, root directory `frontend`
2. Set `VITE_API_URL` to the Space URL

---

# License

MIT License
