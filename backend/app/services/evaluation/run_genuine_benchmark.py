"""
Genuine Empirical Benchmark for ClarifiBids
===========================================
Runs REAL model inferences:
1. Model A: BAAI/bge-small-en-v1.5 (Local HuggingFace weights)
2. Model B: sentence-transformers/all-MiniLM-L6-v2 (Local HuggingFace weights)
3. Lexical: Okapi BM25 (rank_bm25 on tokenized corpus)
4. Hybrid: Dense (BGE-Small) + BM25 using Reciprocal Rank Fusion (k=60)
"""

import os
import sys
import time
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Set offline mode so it doesn't hang checking HF Hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load the actual FAQ dataset from disk or database
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from app.services.ingestion.faq_parser import parse_faq_docx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FAQ_PATH = os.path.join(PROJECT_ROOT, "FAQ.docx")

faq_items = parse_faq_docx(FAQ_PATH)
print(f"Loaded {len(faq_items)} real FAQ chunks from {FAQ_PATH}")

corpus_items = []
for item in faq_items:
    item_num = item.get("metadata", {}).get("item_number", item["chunk_index"])
    corpus_items.append({
        "item_number": int(item_num),
        "question": item["question"],
        "answer": item["answer"],
        "content": item["content"]
    })

corpus_texts = [f"{item['question']} {item['answer']}" for item in corpus_items]
corpus_tokens = [t.lower().split() for t in corpus_texts]

# 1. Initialize BM25
bm25 = BM25Okapi(corpus_tokens)

# 2. Encode corpus with BGE-small
print("Encoding corpus with BAAI/bge-small-en-v1.5...")
t0 = time.time()
bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
bge_corpus_embs = bge_model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
print(f"BGE-small corpus encoded in {time.time() - t0:.2f}s, shape: {bge_corpus_embs.shape}")

# 3. Encode corpus with all-MiniLM-L6-v2
print("Encoding corpus with sentence-transformers/all-MiniLM-L6-v2...")
t0 = time.time()
minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
minilm_corpus_embs = minilm_model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
print(f"MiniLM corpus encoded in {time.time() - t0:.2f}s, shape: {minilm_corpus_embs.shape}")

# Curated evaluation queries with actual FAQ Item # ground truths
EVAL_QUERIES = [
    # Direct
    {"q": "What is the default Date and Time format used in the portal?", "expected": [4]},
    {"q": "What is the validity period of a Digital Signature Certificate?", "expected": [1, 2]},
    {"q": "What browser settings and Java configuration are recommended?", "expected": [8, 43]},
    {"q": "Can a user register multiple accounts with the same DSC token?", "expected": [10, 11]},
    {"q": "What should a user do if they forget their password?", "expected": [12]},
    {"q": "What are the requirements for BoQ financial template upload?", "expected": [29, 30]},
    {"q": "What does EMD fee exemption under MSME require?", "expected": [26, 27]},
    {"q": "What is the maximum file size for document upload?", "expected": [28]},
    {"q": "How does the system ensure bid security and tamper proofing?", "expected": [30]},
    {"q": "Who is eligible to participate as a Foreign Bidder in GePNIC?", "expected": [44, 45]},
    # Procedural / Indirect
    {"q": "DSC token not detected during bid submission error resolution", "expected": [1, 2, 8, 43]},
    {"q": "Procedure to configure foreign bidder DSC and submit bids in foreign currency", "expected": [44, 45, 46]},
    {"q": "Department user assigning tender opener quorum and evaluation members", "expected": [37, 38, 39]},
    {"q": "Step by step procedure to claim EMD exemption and upload certificate", "expected": [26, 27]},
    {"q": "Java blocked by security settings exception site list configuration", "expected": [8, 43]},
    {"q": "Bid resubmission before closing date and overwriting old bid", "expected": [31, 32]},
    {"q": "Corrigendum published after bid submission acknowledging addendum", "expected": [33, 34]},
    {"q": "Updating renewed DSC certificate with same physical USB token", "expected": [2, 11]},
    {"q": "Enabling macros in Excel BoQ template trust center settings", "expected": [29, 30]},
    {"q": "Publishing tender extension corrigendum critical dates modification", "expected": [36, 40]},
    # Multi-hop
    {"q": "Rules connecting DSC token driver, Java JRE version, and BoQ macro encryption", "expected": [1, 8, 29, 30, 43]},
    {"q": "Foreign bidder workflow overseas DSC multi currency BoQ foreign bank guarantee", "expected": [26, 44, 45, 46]},
    {"q": "Two cover technical evaluation and financial cover decryption secrecy", "expected": [26, 28, 30, 38]},
    {"q": "Corrupted BoQ corrigendum re-download and bid re-signing before freeze", "expected": [29, 30, 33, 34]},
    {"q": "DSC revocation check CRL failure and portal TLS security exception", "expected": [1, 2, 8, 43]},
    {"q": "Nodal officer authorization tender opener quorum and technical comparative chart", "expected": [37, 38, 39]},
    {"q": "MSME bidder bid resubmission after BoQ corrigendum attachment verification", "expected": [26, 27, 31, 33]},
    {"q": "User enrollment organization hierarchy mapping and DSC serial number lock", "expected": [10, 11, 14]},
    {"q": "Department user multi-currency BoQ SBI exchange rate comparative evaluation", "expected": [39, 44, 46]},
    {"q": "Expired DSC token driver reinstall Windows certmgr update submission recovery", "expected": [1, 2, 8, 31, 43]}
]

def calculate_metrics(retrieved_items, expected_items, k=5):
    top_k = retrieved_items[:k]
    expected_set = set(expected_items)
    ret_set = set(top_k)
    inter = len(expected_set.intersection(ret_set))
    recall = inter / max(1, len(expected_set))
    precision = inter / max(1, len(top_k))
    
    mrr = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in expected_set:
            mrr = 1.0 / rank
            break
    return recall, precision, mrr

# ----------------- BENCHMARK 1: EMBEDDINGS COMPARISON (TABLE 2) -----------------
methods = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BM25",
    "Hybrid (BM25 + BGE-small RRF)"
]

table2_results = []
K = 5

for method in methods:
    recalls, mrrs, latencies = [], [], []
    
    for item in EVAL_QUERIES:
        q = item["q"]
        expected = item["expected"]
        t_start = time.time()
        
        if method == "BAAI/bge-small-en-v1.5":
            q_emb = bge_model.encode([q], normalize_embeddings=True)[0]
            scores = np.dot(bge_corpus_embs, q_emb)
            ranked_idx = np.argsort(scores)[::-1]
            retrieved = [corpus_items[i]["item_number"] for i in ranked_idx]
            
        elif method == "sentence-transformers/all-MiniLM-L6-v2":
            q_emb = minilm_model.encode([q], normalize_embeddings=True)[0]
            scores = np.dot(minilm_corpus_embs, q_emb)
            ranked_idx = np.argsort(scores)[::-1]
            retrieved = [corpus_items[i]["item_number"] for i in ranked_idx]
            
        elif method == "BM25":
            tokens = q.lower().split()
            scores = bm25.get_scores(tokens)
            ranked_idx = np.argsort(scores)[::-1]
            retrieved = [corpus_items[i]["item_number"] for i in ranked_idx]
            
        elif method == "Hybrid (BM25 + BGE-small RRF)":
            # Dense ranking
            q_emb = bge_model.encode([q], normalize_embeddings=True)[0]
            d_scores = np.dot(bge_corpus_embs, q_emb)
            d_ranks = {corpus_items[i]["item_number"]: r for r, i in enumerate(np.argsort(d_scores)[::-1], 1)}
            
            # BM25 ranking
            tokens = q.lower().split()
            bm_scores = bm25.get_scores(tokens)
            bm_ranks = {corpus_items[i]["item_number"]: r for r, i in enumerate(np.argsort(bm_scores)[::-1], 1)}
            
            # RRF fusion (k=60)
            rrf_scores = {}
            for item_num in d_ranks:
                rrf_scores[item_num] = (1.0 / (60 + d_ranks[item_num])) + (1.0 / (60 + bm_ranks.get(item_num, 999)))
                
            sorted_items = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            retrieved = sorted_items

        elapsed_ms = (time.time() - t_start) * 1000
        latencies.append(elapsed_ms)
        
        rec, prec, mrr = calculate_metrics(retrieved, expected, k=K)
        recalls.append(rec)
        mrrs.append(mrr)
        
    avg_rec = round(float(np.mean(recalls)), 4)
    avg_mrr = round(float(np.mean(mrrs)), 4)
    avg_lat = round(float(np.mean(latencies)), 2)
    
    table2_results.append({
        "Method": method,
        "Recall@5": avg_rec,
        "MRR": avg_mrr,
        "Latency_ms": avg_lat
    })
    print(f"Table 2 Result -> {method}: Recall@5={avg_rec:.4f}, MRR={avg_mrr:.4f}, Latency={avg_lat:.2f}ms")

# Output JSON summary
output_path = os.path.join(os.path.dirname(__file__), "genuine_benchmark_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"table2": table2_results}, f, indent=2)

print(f"\nSaved genuine benchmark results to {output_path}")
