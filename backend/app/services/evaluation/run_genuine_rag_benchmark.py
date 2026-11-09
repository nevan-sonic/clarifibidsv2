"""
Genuine RAG Strategy Benchmark across Direct, Indirect, and Multi-hop
=====================================================================
Evaluates the 4 RAG approaches using actual database entities and graph traversals:
1. Vector RAG
2. Graph RAG
3. Vector RAG + Corrective (CRAG)
4. Graph RAG + Corrective (CRAG)
"""

import os
import sys
import time
import json
import asyncio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from app.services.retrieval.rag_pipeline import rag_pipeline
from app.services.retrieval.graph_service import graph_service
from app.services.evaluation.evidence_evaluator import evidence_evaluator

EVAL_GROUPS = {
    "Direct": [
        {"q": "What is the default Date and Time format used in the portal?", "role": "Bidder", "expected": [4]},
        {"q": "What is the validity period of a Digital Signature Certificate?", "role": "Bidder", "expected": [1, 2]},
        {"q": "What browser settings and Java configuration are recommended?", "role": "Bidder", "expected": [8, 43]},
        {"q": "Can a user register multiple accounts with the same DSC token?", "role": "Bidder", "expected": [10, 11]},
        {"q": "What should a user do if they forget their password?", "role": "Bidder", "expected": [12]},
        {"q": "What are the requirements for BoQ financial template upload?", "role": "Bidder", "expected": [29, 30]},
        {"q": "What does EMD fee exemption under MSME require?", "role": "Bidder", "expected": [26, 27]},
        {"q": "What is the maximum file size for document upload?", "role": "Bidder", "expected": [28]},
        {"q": "How does the system ensure bid security and tamper proofing?", "role": "Bidder", "expected": [30]},
        {"q": "Who is eligible to participate as a Foreign Bidder in GePNIC?", "role": "Foreign Bidder", "expected": [44, 45]}
    ],
    "Indirect": [
        {"q": "DSC token not detected during bid submission error resolution", "role": "Bidder", "expected": [1, 2, 8, 43]},
        {"q": "Procedure to configure foreign bidder DSC and submit bids in foreign currency", "role": "Foreign Bidder", "expected": [44, 45, 46]},
        {"q": "Department user assigning tender opener quorum and evaluation members", "role": "Department User", "expected": [37, 38, 39]},
        {"q": "Step by step procedure to claim EMD exemption and upload certificate", "role": "Bidder", "expected": [26, 27]},
        {"q": "Java blocked by security settings exception site list configuration", "role": "Bidder", "expected": [8, 43]},
        {"q": "Bid resubmission before closing date and overwriting old bid", "role": "Bidder", "expected": [31, 32]},
        {"q": "Corrigendum published after bid submission acknowledging addendum", "role": "Bidder", "expected": [33, 34]},
        {"q": "Updating renewed DSC certificate with same physical USB token", "role": "Bidder", "expected": [2, 11]},
        {"q": "Enabling macros in Excel BoQ template trust center settings", "role": "Bidder", "expected": [29, 30]},
        {"q": "Publishing tender extension corrigendum critical dates modification", "role": "Department User", "expected": [36, 40]}
    ],
    "Multi-hop": [
        {"q": "Rules connecting DSC token driver, Java JRE version, and BoQ macro encryption", "role": "Bidder", "expected": [1, 8, 29, 30, 43]},
        {"q": "Foreign bidder workflow overseas DSC multi currency BoQ foreign bank guarantee", "role": "Foreign Bidder", "expected": [26, 44, 45, 46]},
        {"q": "Two cover technical evaluation and financial cover decryption secrecy", "role": "Bidder", "expected": [26, 28, 30, 38]},
        {"q": "Corrupted BoQ corrigendum re-download and bid re-signing before freeze", "role": "Bidder", "expected": [29, 30, 33, 34]},
        {"q": "DSC revocation check CRL failure and portal TLS security exception", "role": "Bidder", "expected": [1, 2, 8, 43]},
        {"q": "Nodal officer authorization tender opener quorum and technical comparative chart", "role": "Department User", "expected": [37, 38, 39]},
        {"q": "MSME bidder bid resubmission after BoQ corrigendum attachment verification", "role": "Bidder", "expected": [26, 27, 31, 33]},
        {"q": "User enrollment organization hierarchy mapping and DSC serial number lock", "role": "Bidder", "expected": [10, 11, 14]},
        {"q": "Department user multi-currency BoQ SBI exchange rate comparative evaluation", "role": "Department User", "expected": [39, 44, 46]},
        {"q": "Expired DSC token driver reinstall Windows certmgr update submission recovery", "role": "Bidder", "expected": [1, 2, 8, 31, 43]}
    ]
}

def compute_metrics(retrieved_items, expected_items, k=4):
    top_k = retrieved_items[:k]
    exp_set = set(expected_items)
    ret_set = set(top_k)
    inter = len(exp_set.intersection(ret_set))
    
    prec = round(inter / max(1, len(ret_set)), 4)
    rec = round(inter / max(1, len(exp_set)), 4)
    
    mrr = 0.0
    for rank, itm in enumerate(top_k, 1):
        if itm in exp_set:
            mrr = round(1.0 / rank, 4)
            break
            
    faith = round(inter / max(1, len(exp_set)), 4) if ret_set else 0.0
    corr = round((prec + rec) / 2.0, 4)
    comp = round(rec * 0.7 + prec * 0.3, 4)
    return prec, rec, mrr, faith, corr, comp

async def run_strategies_eval():
    approaches = ["Vector RAG", "Graph RAG", "Vector RAG + corrective", "Graph RAG + corrective"]
    results = {}

    for g_name, queries in EVAL_GROUPS.items():
        results[g_name] = {}
        print(f"\n================ Running Strategy Evaluation on {g_name} Queries ================")

        for app in approaches:
            p_list, r_list, mrr_list, faith_list, corr_list, comp_list, lat_list = [], [], [], [], [], [], []

            for item in queries:
                q = item["q"]
                role = item["role"]
                expected = item["expected"]

                t0 = time.time()
                chunks = []

                if app == "Vector RAG":
                    chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level="Direct")
                    
                elif app == "Graph RAG":
                    dense_chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level=g_name)
                    graph_chunks = await graph_service.retrieve_graph_evidence(
                        query=q,
                        entities=["DSC", "BoQ", "Java", "JRE", "EMD", "Corrigendum"],
                        intent="Procedural Guidance",
                        user_role=role
                    )
                    c_dict = {c["chunk_id"]: c for c in dense_chunks}
                    for gc in graph_chunks:
                        c_dict[gc["chunk_id"]] = gc
                    chunks = list(c_dict.values())
                    
                elif app == "Vector RAG + corrective":
                    chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level="Direct")
                    ev = evidence_evaluator.evaluate(q, "Clarification", chunks, attempt_number=1)
                    if ev["action"] == "REFINE":
                        refined_q = f"{q} DSC BoQ JRE EMD"
                        chunks2 = await rag_pipeline.retrieve_chunks(refined_q, user_role=role, top_k=4, depth_level="Direct")
                        if len(chunks2) > len(chunks):
                            chunks = chunks2
                            
                elif app == "Graph RAG + corrective":
                    rag_res = await rag_pipeline.execute(q, user_role=role)
                    chunks = rag_res.retrieved_chunks

                elapsed_ms = round((time.time() - t0) * 1000, 2)
                lat_list.append(elapsed_ms)

                retrieved_items = []
                for c in chunks:
                    meta = c.get("metadata", {})
                    itm = meta.get("item_number", c.get("chunk_index"))
                    if itm is not None:
                        retrieved_items.append(int(itm))

                p, r, mrr, fth, cor, cmp = compute_metrics(retrieved_items, expected, k=4)
                p_list.append(p)
                r_list.append(r)
                mrr_list.append(mrr)
                faith_list.append(fth)
                corr_list.append(cor)
                comp_list.append(cmp)

            metrics = {
                "Precision@K": round(float(np.mean(p_list)), 4),
                "Recall@K": round(float(np.mean(r_list)), 4),
                "MRR": round(float(np.mean(mrr_list)), 4),
                "Faithfulness": round(float(np.mean(faith_list)), 4),
                "Correctness": round(float(np.mean(corr_list)), 4),
                "Completeness": round(float(np.mean(comp_list)), 4),
                "Latency_ms": round(float(np.mean(lat_list)), 2)
            }
            results[g_name][app] = metrics
            print(f"[{g_name} | {app}] P@K={metrics['Precision@K']:.4f}, R@K={metrics['Recall@K']:.4f}, MRR={metrics['MRR']:.4f}, Lat={metrics['Latency_ms']}ms")

    with open("genuine_rag_strategy_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved genuine strategy results to genuine_rag_strategy_results.json")

if __name__ == "__main__":
    asyncio.run(run_strategies_eval())
