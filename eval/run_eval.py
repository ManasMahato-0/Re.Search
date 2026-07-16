"""
Eval harness: run labeled queries through the search pipeline, score results.

Usage (from backend/ so index paths resolve):
    cd backend
    python ../eval/run_eval.py                 # run + score
    python ../eval/run_eval.py --tag baseline  # save scores under a tag

Metrics:
    nDCG@10   - ranking quality of final (reranked) results
    MRR@10    - reciprocal rank of first relevant result
    recall@30 - did relevant doc survive retrieval+fusion (pre-rerank pool)

Imports the pipeline directly from main.py — no server needed.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(EVAL_DIR, "..", "backend")
QUERIES_FILE = os.path.join(EVAL_DIR, "queries.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

K_FINAL = 10   # depth for nDCG / MRR


def normalize_url(url: str) -> str:
    """Match URLs regardless of scheme, trailing slash, or fragment."""
    url = url.strip().lower()
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return url.rstrip("/").split("#")[0]


def load_queries():
    queries = []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def ndcg_at_k(result_urls, relevant, k):
    """Binary relevance nDCG."""
    dcg = 0.0
    for rank, url in enumerate(result_urls[:k]):
        if url in relevant:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(result_urls, relevant, k):
    for rank, url in enumerate(result_urls[:k]):
        if url in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def recall(candidate_urls, relevant):
    if not relevant:
        return 0.0
    hits = sum(1 for u in relevant if u in candidate_urls)
    return hits / len(relevant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="label for this run (e.g. baseline)")
    args = parser.parse_args()

    # Import the pipeline from backend/main.py (loads indexes + models once)
    sys.path.insert(0, BACKEND_DIR)
    os.chdir(BACKEND_DIR)  # index files are opened with relative paths
    print("Loading search pipeline (indexes + models)...")
    from main import run_search, RERANK_POOL  # noqa: E402

    queries = load_queries()
    print(f"Running {len(queries)} queries...\n")

    per_query = []
    for i, q in enumerate(queries, 1):
        relevant = {normalize_url(u) for u in q["relevant_urls"]}
        results, candidate_urls = run_search(q["query"], final_k=K_FINAL)

        result_urls = [normalize_url(r["url"]) for r in results]
        cand_urls = {normalize_url(u) for u in candidate_urls}

        row = {
            "qid": q["qid"],
            "query": q["query"],
            "ndcg": ndcg_at_k(result_urls, relevant, K_FINAL),
            "mrr": mrr_at_k(result_urls, relevant, K_FINAL),
            "recall": recall(cand_urls, relevant),
            "top_url": results[0]["url"] if results else None,
        }
        per_query.append(row)

        marker = " " if row["mrr"] > 0 else ("~" if row["recall"] > 0 else "X")
        print(f"[{marker}] {i:>2}/{len(queries)} {q['qid']:<10} "
              f"ndcg={row['ndcg']:.2f} mrr={row['mrr']:.2f} recall={row['recall']:.0f}  {q['query'][:55]}")

    n = len(per_query)
    summary = {
        f"nDCG@{K_FINAL}": sum(r["ndcg"] for r in per_query) / n,
        f"MRR@{K_FINAL}": sum(r["mrr"] for r in per_query) / n,
        f"recall@{RERANK_POOL}": sum(r["recall"] for r in per_query) / n,
    }

    print("\n" + "=" * 52)
    print(f"{'metric':<12} {'score':>8}    ({n} queries)")
    print("-" * 52)
    for metric, score in summary.items():
        print(f"{metric:<12} {score:>8.4f}")
    print("=" * 52)

    # Failure buckets
    missed_retrieval = [r["qid"] for r in per_query if r["recall"] == 0]
    missed_ranking = [r["qid"] for r in per_query if r["recall"] > 0 and r["mrr"] == 0]
    if missed_retrieval:
        print(f"\nX  retrieval misses (doc never in candidate pool): {missed_retrieval}")
    if missed_ranking:
        print(f"~  ranking misses (retrieved but reranked out of top {K_FINAL}): {missed_ranking}")

    # Persist run
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{stamp}_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "timestamp": stamp, "summary": summary,
                   "per_query": per_query}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
