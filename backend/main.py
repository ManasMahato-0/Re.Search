
import pickle
import re
import os
import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from query_expansion import expand_query


from bm25 import BM25, tokenize  

# ---------------------------------------------------------
# TUNING KNOBS
# ---------------------------------------------------------
TOP_K_POOL = 50       # candidates pulled from EACH engine before fusion
MIN_COSINE_SIM = float(os.environ.get("MIN_COSINE_SIM", 0.4))
                       # unrelated and dropped
RRF_K = 60             # standard RRF damping constant
BM25_WEIGHT = 1.2      # lexical hits weighted slightly above semantic ones —
DENSE_WEIGHT = 1.0     # for doc/paper search, exact keyword match is usually
EXPANSION_WEIGHT = 0.4 # ranked lists from LLM query variants count this much
                       # relative to the original query's lists
QUERY_EXPANSION =os.environ.get("QUERY_EXPANSION", "0") == "1"
                       # the stronger signal; tune per taste.
RERANK_POOL = 30      # how many fused docs the cross-encoder rescores
FINAL_RESULTS = 10     # how many results the API returns
EXPANSION_TRIGGER = 999

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

bm25_engine.build_postings()  # inverted index for sparse scoring

faiss_index = faiss.read_index("vector_index.faiss")



model = SentenceTransformer("BAAI/bge-small-en-v1.5")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", model_kwargs={"dtype": "float16"})


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
# SEARCH PIPELINE (shared by the API endpoint and eval harness)
# ---------------------------------------------------------
def run_search(q: str, final_k: int = FINAL_RESULTS):
    
        # Build query list: original + (optionally) LLM expansions
    queries = [(q, 1.0)]
    if QUERY_EXPANSION:
        top_bm25 = max(bm25_engine.get_scores_sparse(q).values(), default=0.0)
        if top_bm25 < EXPANSION_TRIGGER:   # weak lexical match 
            queries += [(v, EXPANSION_WEIGHT) for v in expand_query(q)]


    
    rrf_scores = {}
    for query_text, weight in queries:
        # LEXICAL: BM25 via inverted index (only chunks containing query terms)
        sparse = bm25_engine.get_scores_sparse(query_text)
        bm25_ranked = sorted(sparse, key=sparse.get, reverse=True)[:TOP_K_POOL]

        # SEMANTIC: FAISS cosine search
        query_vector = model.encode(
            [BGE_QUERY_PREFIX + query_text], normalize_embeddings=True
        ).astype("float32")
        k_search = min(TOP_K_POOL, faiss_index.ntotal)
        similarities, indices = faiss_index.search(query_vector, k_search)
        faiss_ranked = [
            int(idx)
            for sim, idx in zip(similarities[0], indices[0])
            if idx != -1 and sim >= MIN_COSINE_SIM
        ]

        # Weighted RRF accumulation across all variants
        for rank, chunk_idx in enumerate(bm25_ranked):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + weight * BM25_WEIGHT / (RRF_K + rank + 1)
        for rank, chunk_idx in enumerate(faiss_ranked):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + weight * DENSE_WEIGHT / (RRF_K + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


    # 4. COLLAPSE chunks
    doc_best_chunk = {}
    for chunk_idx, rrf_score in fused:
        doc_idx = chunk_to_doc[chunk_idx]
        if doc_idx not in doc_best_chunk:
            doc_best_chunk[doc_idx] = (chunk_idx, rrf_score)
        if len(doc_best_chunk) >= RERANK_POOL:
            break  
    candidates = [
        {"doc_idx": d, "chunk_idx": c, "rrf_score": s}
        for d, (c, s) in doc_best_chunk.items()
    ]
    candidate_urls = [documents[c["doc_idx"]].get("url", "#") for c in candidates]

    if not candidates:
        return [], []

    # 5. RERANK with the cross-encoder
    pairs = [
        (q, f"{documents[c['doc_idx']].get('title', '')}. {chunk_texts[c['chunk_idx']]}")
        for c in candidates
    ]
    rerank_scores = reranker.predict(pairs)
    for cand, score in zip(candidates, rerank_scores):
        cand["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # 6. BUILD the response
    top_results = []
    for cand in candidates[:final_k]:
        doc = documents[cand["doc_idx"]]
        chunk_text = chunk_texts[cand["chunk_idx"]]
        top_results.append({
            "title": doc.get("title", "Untitled Document"),
            "url": doc.get("url", "#"),
            "snippet": make_snippet(chunk_text, q),
            "score": round(cand["rerank_score"], 4),
        })

    return top_results, candidate_urls


# ---------------------------------------------------------
# SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    top_results, _ = run_search(q)
    return {
        "status": "success",
        "query": q,
        "results": top_results,
    }
