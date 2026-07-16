

import os
import json
import math
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from bm25 import BM25

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_FILES = [
    "data/research_dataset.json",   
    "data/crawler_dataset.jsonl",  
]


CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# How many times to repeat the title in the BM25 text of each chunk.
TITLE_WEIGHT = 3

MODEL_NAME = "BAAI/bge-small-en-v1.5"
import torch
EMBED_BATCH_SIZE = 256 if torch.cuda.is_available() else 64


import re

# Nav/boilerplate phrases
NAV_PHRASES = [
    "skip to main content", "skip to content", "skip to search",
    "skip navigation links", "skip to docs navigation",
    "table of contents", "edit on github", "view on github",
    "create account", "log in", "sign in", "toggle navigation",
    "light mode", "dark mode", "back to top",
    "read in english", "keyboard shortcuts",
]

def clean_boilerplate(text: str, title: str) -> str:
    t = text
    tl = title.strip()
    while tl and t.lower().startswith(tl.lower()):
        t = t[len(tl):].lstrip(" |-·—>:.")
    for phrase in NAV_PHRASES:
        t = re.sub(re.escape(phrase), " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------
# 1. LOAD + DEDUPLICATE DOCUMENTS
# ---------------------------------------------------------
print("Loading data from datasets...")

documents = []      # one entry per SOURCE DOCUMENT 
seen_urls = set()   # dedup: same URL crawled twice -> keep first occurrence

for file_path in DATA_FILES:
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        continue

    print(f"Loading: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue  
            url = doc.get("url", "#")
            text = clean_boilerplate(doc.get("text_content", "").strip(), doc.get("title", ""))

            # Skip near-empty pages (nav-only pages, error pages, etc.)
            if len(text) < 200:
                continue
            # Skip duplicates of a URL we've already ingested
            if url in seen_urls:
                continue
            seen_urls.add(url)

            documents.append({
                "title": doc.get("title", "Untitled Document"),
                "url": url,
                "text_content": text,  
            })

print(f"Loaded {len(documents)} unique documents.")


# ---------------------------------------------------------
# 2. CHUNK DOCUMENTS
# ---------------------------------------------------------

print("Chunking documents...")

chunk_texts = []   
bm25_texts = []     
chunk_to_doc = []   

for doc_idx, doc in enumerate(documents):
    words = doc["text_content"].split()
    title = doc["title"]

    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, max(len(words), 1), step):
        chunk_words = words[start : start + CHUNK_SIZE]
        if not chunk_words:
            break
        body = " ".join(chunk_words)

       
        chunk_texts.append(f"{title}. {body}")
        bm25_texts.append(f"{(title + ' ') * TITLE_WEIGHT}{body}")
        chunk_to_doc.append(doc_idx)

        if start + CHUNK_SIZE >= len(words):
            break

print(f"Created {len(chunk_texts)} chunks from {len(documents)} documents.")


# ---------------------------------------------------------
# 3. BUILD BM25 LEXICAL INDEX (over chunks)
# ---------------------------------------------------------
print("Building BM25 lexical index...")
bm25_engine = BM25(bm25_texts)

print("Saving bm25_index.pkl ...")
with open("bm25_index.pkl", "wb") as f:
    pickle.dump({
        "bm25": bm25_engine,
        "documents": documents,
        "chunk_to_doc": chunk_to_doc,
        "chunk_texts": chunk_texts,  
    }, f)


# ---------------------------------------------------------
# 4. BUILD FAISS DENSE VECTOR INDEX (over the same chunks)
# ---------------------------------------------------------
print(f"Loading embedding model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)
print(f"Embedding device: {model.device} (batch size {EMBED_BATCH_SIZE})")


dimension = model.get_embedding_dimension() 
faiss_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
faiss_index.hnsw.efSearch = 64


print("Embedding chunks (batched to keep memory flat)...")
total_batches = math.ceil(len(chunk_texts) / EMBED_BATCH_SIZE)

for i in range(0, len(chunk_texts), EMBED_BATCH_SIZE):
    batch = chunk_texts[i : i + EMBED_BATCH_SIZE]

    vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    faiss_index.add(np.asarray(vectors, dtype="float32"))

    batch_num = i // EMBED_BATCH_SIZE + 1
    if batch_num % 50 == 0 or batch_num == total_batches:
        print(f"   -> embedded batch {batch_num}/{total_batches}")

print("Saving vector_index.faiss ...")
faiss.write_index(faiss_index, "vector_index.faiss")

print("\nINDEXING COMPLETE — restart the FastAPI server to pick up the new index.")
