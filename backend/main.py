
import pickle
import re

import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder


from bm25 import BM25, tokenize  

# ---------------------------------------------------------
# TUNING KNOBS
# ---------------------------------------------------------
TOP_K_POOL = 50        # candidates pulled from EACH engine before fusion
MIN_COSINE_SIM = 0.4   # FAISS hits below this similarity are considered
                       # unrelated and dropped
RRF_K = 60             # standard RRF damping constant
BM25_WEIGHT = 1.2      # lexical hits weighted slightly above semantic ones —
DENSE_WEIGHT = 1.0     # for doc/paper search, exact keyword match is usually
                       # the stronger signal; tune per taste.
RERANK_POOL = 30       # how many fused docs the cross-encoder rescores
FINAL_RESULTS = 10     # how many results the API returns

# ---------------------------------------------------------
# SERVER INITIALIZATION & INDEX LOADING
# ---------------------------------------------------------
app = FastAPI(title="Hybrid Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting server and loading indices into memory...")

with open("bm25_index.pkl", "rb") as f:
    payload = pickle.load(f)
    bm25_engine = payload["bm25"]
    documents = payload["documents"]       
    chunk_to_doc = payload["chunk_to_doc"] 
    chunk_texts = payload["chunk_texts"]    

faiss_index = faiss.read_index("vector_index.faiss")


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Server initialization complete. Ready to serve traffic.")


# ---------------------------------------------------------
# RECIPROCAL RANK FUSION (RRF)
# ---------------------------------------------------------
def reciprocal_rank_fusion(bm25_rankings, faiss_rankings, k=RRF_K):
    
    rrf_scores = {}

    for rank, chunk_idx in enumerate(bm25_rankings):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + BM25_WEIGHT / (k + rank + 1)


    for rank, chunk_idx in enumerate(faiss_rankings):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + DENSE_WEIGHT / (k + rank + 1)

    # Highest combined score first
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------
# QUERY-AWARE SNIPPETS
# ---------------------------------------------------------
def make_snippet(chunk_text: str, query: str, width: int = 200) -> str:
    """
    Show the user the part of the chunk that actually matched the query,
    instead of blindly taking the first 200 characters (which was usually
    the page header / nav text and looked identical for every result).

    Strategy: find the first occurrence of any query word in the chunk and
    center a `width`-character window on it.
    """
    query_words = tokenize(query)
    lower_text = chunk_text.lower()

    # Position of the earliest query-word occurrence (whole-word match)
    best_pos = -1
    for word in query_words:
        m = re.search(rf"\b{re.escape(word)}\b", lower_text)
        if m and (best_pos == -1 or m.start() < best_pos):
            best_pos = m.start()

    if best_pos == -1:
        snippet = chunk_text[:width]
        return snippet + ("..." if len(chunk_text) > width else "")

    start = max(0, best_pos - width // 3)
    end = min(len(chunk_text), start + width)

    if start > 0:
        space = chunk_text.find(" ", start)
        if space != -1 and space < end:
            start = space + 1
    snippet = chunk_text[start:end].strip()

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(chunk_text) else ""
    return f"{prefix}{snippet}{suffix}"


# ---------------------------------------------------------
# SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    # 1. LEXICAL: BM25 over all chunks ---
    bm25_scores = bm25_engine.get_scores(q)
    valid = [i for i, s in enumerate(bm25_scores) if s > 0]
    bm25_ranked = sorted(valid, key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_POOL]

    # 2. SEMANTIC: FAISS cosine search ---
    
    query_vector = model.encode(
        [BGE_QUERY_PREFIX + q], normalize_embeddings=True
    ).astype("float32")
    k_search = min(TOP_K_POOL, faiss_index.ntotal)
    similarities, indices = faiss_index.search(query_vector, k_search)

    # Similarity cutoff
    faiss_ranked = [
        int(idx)
        for sim, idx in zip(similarities[0], indices[0])
        if idx != -1 and sim >= MIN_COSINE_SIM
    ]

    # 3. FUSE the two ranked lists  ---
    fused = reciprocal_rank_fusion(bm25_ranked, faiss_ranked)

    # 4. COLLAPSE chunks 
    doc_best_chunk = {}   
    for chunk_idx, rrf_score in fused:
        doc_idx = chunk_to_doc[chunk_idx]
        if doc_idx not in doc_best_chunk:
            doc_best_chunk[doc_idx] = (chunk_idx, rrf_score)
        if len(doc_best_chunk) >= RERANK_POOL:
            break  # we have enough candidates for the reranker

    candidates = [
        {"doc_idx": d, "chunk_idx": c, "rrf_score": s}
        for d, (c, s) in doc_best_chunk.items()
    ]

    if not candidates:
        return {"status": "success", "query": q, "results": []}

    # 5. RERANK with the cross-encoder 
    pairs = [(q, chunk_texts[c["chunk_idx"]]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for cand, score in zip(candidates, rerank_scores):
        cand["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # 6. BUILD the response 
    top_results = []
    for cand in candidates[:FINAL_RESULTS]:
        doc = documents[cand["doc_idx"]]
        chunk_text = chunk_texts[cand["chunk_idx"]]
        top_results.append({
            "title": doc.get("title", "Untitled Document"),
            "url": doc.get("url", "#"),
            "snippet": make_snippet(chunk_text, q),
            "score": round(cand["rerank_score"], 4),
        })

    return {
        "status": "success",
        "query": q,
        "results": top_results,
    }
