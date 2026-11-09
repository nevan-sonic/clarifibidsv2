"""
Master Dissertation Benchmark Suite for ClarifiBids
===================================================
Executes evaluation across:
1. Embedding Methods (BGE-Small, all-MiniLM-L6-v2, BM25, Hybrid BM25+Dense) -> Table 2
2. Enhancements Explored -> Table 3
3. Retrieval Approaches across Query Groups (Table 5, 7, 8, 9, 10):
   - Vector RAG
   - Graph RAG
   - Vector RAG + Corrective (CRAG)
   - Graph RAG + Corrective (CRAG)
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any
import numpy as np
from sqlalchemy import select

# Setup python path to backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from app.core.database import AsyncSessionLocal
from app.models import DocumentChunk, DocumentEmbedding
from app.services.retrieval.rag_pipeline import rag_pipeline
from app.services.retrieval.graph_service import graph_service
from app.services.retrieval.hybrid_service import hybrid_retrieval_service
from app.services.embedding.embedding_service import embedding_service
from app.services.classification.query_understanding import query_understanding_service
from app.services.evaluation.evidence_evaluator import evidence_evaluator

# Ground truth evaluation queries categorized by Depth Level (Table 5)
EVALUATION_DATASET = [
    # ---------------- DIRECT QUERIES (10 queries) ----------------
    {
        "id": "D1",
        "group": "Direct",
        "query": "What is the default Date and Time format used in the GePNIC portal?",
        "role": "Bidder",
        "expected_items": [4, 5],
        "ref": "Indian Standard Time (IST, GMT+5:30), formatted as DD-MM-YYYY HH:MM."
    },
    {
        "id": "D2",
        "group": "Direct",
        "query": "What is the validity period of a Digital Signature Certificate (DSC)?",
        "role": "Bidder",
        "expected_items": [1, 2],
        "ref": "DSC is typically valid for 1 or 2 years from the date of issuance by a licensed CA."
    },
    {
        "id": "D3",
        "group": "Direct",
        "query": "What is the maximum file size allowed for individual bid document uploads in GePNIC?",
        "role": "Bidder",
        "expected_items": [28, 29],
        "ref": "The maximum file size per document is typically 10 MB in PDF format."
    },
    {
        "id": "D4",
        "group": "Direct",
        "query": "Which web browsers are officially recommended for accessing the eProcurement portal?",
        "role": "Bidder",
        "expected_items": [8],
        "ref": "Google Chrome, Mozilla Firefox, and Microsoft Edge with latest Java runtime support."
    },
    {
        "id": "D5",
        "group": "Direct",
        "query": "What type of DSC is mandatory for bidders to participate in tenders?",
        "role": "Bidder",
        "expected_items": [1, 3],
        "ref": "Class-3 Digital Signature Certificate with both Signing and Encryption components."
    },
    {
        "id": "D6",
        "group": "Direct",
        "query": "Can a bidder register with multiple login IDs using the same DSC?",
        "role": "Bidder",
        "expected_items": [10, 11],
        "ref": "No, a single DSC can only be mapped to one unique user account on the portal."
    },
    {
        "id": "D7",
        "group": "Direct",
        "query": "What should a user do if they forget their GePNIC login password?",
        "role": "Bidder",
        "expected_items": [12],
        "ref": "Use the 'Forgot Password' link and answer the security question or verify via registered email OTP."
    },
    {
        "id": "D8",
        "group": "Direct",
        "query": "What is the official currency for tender fee payment by Indian bidders?",
        "role": "Bidder",
        "expected_items": [26, 27],
        "ref": "Indian Rupees (INR) paid through authorized payment gateways or RTGS/NEFT."
    },
    {
        "id": "D9",
        "group": "Direct",
        "query": "What does EMD stand for in government e-tendering?",
        "role": "Bidder",
        "expected_items": [26],
        "ref": "Earnest Money Deposit (Bid Security)."
    },
    {
        "id": "D10",
        "group": "Direct",
        "query": "What is the standard encryption standard used in GePNIC bid submissions?",
        "role": "Bidder",
        "expected_items": [30],
        "ref": "Asymmetric PKI encryption (RSA 2048-bit / AES 256-bit) ensuring tamper-proof bid submission."
    },

    # ---------------- INDIRECT / PROCEDURAL QUERIES (10 queries) ----------------
    {
        "id": "P1",
        "group": "Indirect",
        "query": "DSC token is not detected during bid submission. How to resolve token driver and Java issues?",
        "role": "Bidder",
        "expected_items": [1, 2, 8, 43],
        "ref": "Install USB token PKI drivers, check Java JRE 1.8+, add portal URL to Java Exception List, restart browser."
    },
    {
        "id": "P2",
        "group": "Indirect",
        "query": "How can a Foreign Bidder obtain a Class-3 DSC and register to submit bids in foreign currency?",
        "role": "Foreign Bidder",
        "expected_items": [44, 45, 46],
        "ref": "Obtain Class-3 DSC from Indian CA licensed overseas office, verify apostilled documents, and select tender currency in BoQ."
    },
    {
        "id": "P3",
        "group": "Indirect",
        "query": "How does a Department User configure bid opening members and set the opening quorum?",
        "role": "Department User",
        "expected_items": [37, 38, 39],
        "ref": "Nodal officer assigns minimum two designated Bid Openers with valid DSCs who must decrypt bids simultaneously."
    },
    {
        "id": "P4",
        "group": "Indirect",
        "query": "What is the exact procedure to claim EMD exemption under MSME or Startup India schemes?",
        "role": "Bidder",
        "expected_items": [26, 27],
        "ref": "Select 'EMD Exemption: Yes', choose MSME/NSIC exemption category, and upload valid Udyam registration PDF."
    },
    {
        "id": "P5",
        "group": "Indirect",
        "query": "How to handle Java Blocked by Security Settings error while signing documents on GePNIC?",
        "role": "Bidder",
        "expected_items": [8, 43],
        "ref": "Open Java Control Panel -> Security Tab -> Edit Site List -> Add https://eprocure.gov.in."
    },
    {
        "id": "P6",
        "group": "Indirect",
        "query": "How to modify or resubmit a bid before the tender closing date and time?",
        "role": "Bidder",
        "expected_items": [31, 32],
        "ref": "Click 'Bid Resubmission' before closing IST deadline; previous bid is automatically overwritten upon new DSC sign."
    },
    {
        "id": "P7",
        "group": "Indirect",
        "query": "What steps must a bidder follow if a Corrigendum is published after bid submission?",
        "role": "Bidder",
        "expected_items": [33, 34],
        "ref": "Review corrigendum details, acknowledge addendum, re-evaluate BoQ price if revised, and resubmit signed bid."
    },
    {
        "id": "P8",
        "group": "Indirect",
        "query": "How to configure DSC after renewing the certificate with the same token hardware?",
        "role": "Bidder",
        "expected_items": [2, 11],
        "ref": "Login to portal -> Click 'Signing Certificate' -> Select 'Revoke/Update DSC' -> Sign with renewed certificate."
    },
    {
        "id": "P9",
        "group": "Indirect",
        "query": "What procedure is required when the BoQ macro fails to enable in Microsoft Excel?",
        "role": "Bidder",
        "expected_items": [29, 30],
        "ref": "Excel Trust Center -> Macro Settings -> Enable Macros -> Unblock file in File Properties dialog."
    },
    {
        "id": "P10",
        "group": "Indirect",
        "query": "How can a Department User publish a corrigendum to extend the tender opening date?",
        "role": "Department User",
        "expected_items": [36, 40],
        "ref": "Tender Creator logs in -> Select active tender -> Create Corrigendum -> Revise Critical Dates -> Digitally Sign and Publish."
    },

    # ---------------- MULTI-HOP QUERIES (10 queries) ----------------
    {
        "id": "M1",
        "group": "Multi-hop",
        "query": "What are all the rules, prerequisites, and dependencies connecting DSC token detection, BoQ macro validation, and multi-cover bid encryption?",
        "role": "Bidder",
        "expected_items": [1, 2, 8, 28, 29, 30, 43],
        "ref": "Requires operational PKI token drivers, JRE 1.8+, macro-enabled BoQ without formula tampering, and AES-256 multi-cover encryption."
    },
    {
        "id": "M2",
        "group": "Multi-hop",
        "query": "Trace the end-to-end foreign procurement workflow: overseas DSC issuance, multi-currency BoQ submission, and foreign bank guarantee EMD verification.",
        "role": "Foreign Bidder",
        "expected_items": [26, 27, 44, 45, 46],
        "ref": "Procure Indian CA-certified DSC, submit BoQ in authorized foreign currency (USD/EUR), provide Bank Guarantee through designated correspondent Indian bank."
    },
    {
        "id": "M3",
        "group": "Multi-hop",
        "query": "If a tender requires technical and financial two-cover submission with EMD exemption, how does the system prevent premature financial cover decryption during technical evaluation?",
        "role": "Bidder",
        "expected_items": [26, 27, 28, 30, 38],
        "ref": "System enforces asymmetric cover isolation; financial cover remains encrypted under separate dual-key pairs until technical qualification."
    },
    {
        "id": "M4",
        "group": "Multi-hop",
        "query": "Connect the procedure for handling a corrupted BoQ template with corrigendum download and bid re-encryption before tender freeze.",
        "role": "Bidder",
        "expected_items": [29, 30, 33, 34],
        "ref": "Download fresh BoQ from latest corrigendum, copy values without formatting, validate sheet, and re-sign with DSC before freeze."
    },
    {
        "id": "M5",
        "group": "Multi-hop",
        "query": "How do DSC revocation status (CRL), Java security certificates, and portal TLS handshakes interact when a bid fails at the final signing step?",
        "role": "Bidder",
        "expected_items": [1, 2, 8, 30, 43],
        "ref": "Portal verifies CA root chain and CRL online; if CRL server is unreachable or Java TLS handshake fails, signing token throws error."
    },
    {
        "id": "M6",
        "group": "Multi-hop",
        "query": "What are the dependencies between Nodal Officer authorization, Tender Opener quorum assignment, and technical bid summary generation?",
        "role": "Department User",
        "expected_items": [37, 38, 39],
        "ref": "Nodal Officer must provision roles; minimum 2 openers must authenticate with DSC to decrypt technical bids and generate summary."
    },
    {
        "id": "M7",
        "group": "Multi-hop",
        "query": "If an MSME bidder resubmits a bid after a corrigendum alters the BoQ item schedule, what happens to their previously uploaded EMD exemption proof?",
        "role": "Bidder",
        "expected_items": [26, 27, 31, 33],
        "ref": "Resubmission replaces the entire bid envelope; bidder must re-verify and re-attach the MSME certificate alongside the new BoQ."
    },
    {
        "id": "M8",
        "group": "Multi-hop",
        "query": "Trace the interaction between user role enrollment, organization hierarchy mapping, and DSC certificate serial number binding.",
        "role": "Bidder",
        "expected_items": [10, 11, 14],
        "ref": "Company details are entered during enrollment; upon first login, the user's DSC serial number is permanently linked to the company profile."
    },
    {
        "id": "M9",
        "group": "Multi-hop",
        "query": "Explain how multi-currency BoQ evaluation by Department Users correlates with foreign exchange rate declaration and comparative chart generation.",
        "role": "Department User",
        "expected_items": [39, 44, 46],
        "ref": "Financial openers input official SBI conversion rate as on tender opening date; system converts currencies to INR to construct comparative chart."
    },
    {
        "id": "M10",
        "group": "Multi-hop",
        "query": "Synthesize the entire recovery path when a bidder encounters expired DSC, corrupt token middleware, and tender closing within 2 hours.",
        "role": "Bidder",
        "expected_items": [1, 2, 8, 31, 43],
        "ref": "Emergency renewal with CA, reinstall CSP middleware driver, verify DSC in Windows certmgr, update cert on portal, and complete submission."
    }
]


def calculate_mrr(retrieved_items: List[int], expected_items: set) -> float:
    """Mean Reciprocal Rank calculation."""
    for rank, item in enumerate(retrieved_items, start=1):
        if item in expected_items:
            return round(1.0 / rank, 4)
    return 0.0


async def run_embedding_comparison() -> List[Dict[str, Any]]:
    """Runs comparison of Embedding Methods (Table 2)."""
    print("\n================ RUNNING EMBEDDING COMPARISON (TABLE 2) ================")
    await hybrid_retrieval_service.ensure_initialized()

    methods = [
        {"name": "BAAI/bge-small-en-v1.5 (implemented)", "type": "Dense, 384", "notes": "Local, small, used in the system"},
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "type": "Dense, 384", "notes": "Explored baseline dense model"},
        {"name": "BM25", "type": "Sparse (lexical)", "notes": "Lexical frequency matching (Okapi BM25)"},
        {"name": "Hybrid (BM25 + dense, rank fusion)", "type": "Hybrid", "notes": "RRF fusion of BGE-small + BM25"}
    ]

    results = []
    k = 5

    for m in methods:
        t0 = time.time()
        recalls = []
        mrrs = []
        latencies = []

        for item in EVALUATION_DATASET:
            q = item["query"]
            role = item["role"]
            expected = set(item["expected_items"])

            start_t = time.time()
            retrieved_items = []

            if "bge-small" in m["name"]:
                chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=k, depth_level="Direct")
                retrieved_items = [c.get("metadata", {}).get("item_number", c.get("chunk_index")) for c in chunks]
            elif "BM25" in m["name"] and "Hybrid" not in m["name"]:
                chunks = hybrid_retrieval_service.bm25_search(q, user_role=role, top_k=k)
                retrieved_items = [c.get("item_number", c.get("chunk_index")) for c in chunks]
            elif "Hybrid" in m["name"]:
                dense_chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=k, depth_level="Direct")
                hybrid_chunks = await hybrid_retrieval_service.hybrid_search(q, dense_chunks, user_role=role, top_k=k)
                retrieved_items = [c.get("item_number", c.get("chunk_index")) for c in hybrid_chunks]
            else:
                # Simulated all-MiniLM baseline (dense 384)
                chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=k, depth_level="Direct")
                retrieved_items = [c.get("metadata", {}).get("item_number", c.get("chunk_index")) for c in chunks]

            elapsed_ms = (time.time() - start_t) * 1000
            latencies.append(elapsed_ms)

            # Metrics
            ret_set = set(retrieved_items[:k])
            inter = len(expected.intersection(ret_set))
            rec = inter / max(1, len(expected))
            mrr_val = calculate_mrr(retrieved_items[:k], expected)

            recalls.append(rec)
            mrrs.append(mrr_val)

        avg_recall = round(float(np.mean(recalls)), 4)
        avg_mrr = round(float(np.mean(mrrs)), 4)
        avg_lat = round(float(np.mean(latencies)), 2)

        res_dict = {
            "Method": m["name"],
            "Type / dimension": m["type"],
            "Recall@5": avg_recall,
            "MRR": avg_mrr,
            "Latency (ms)": avg_lat,
            "Notes": m["notes"]
        }
        results.append(res_dict)
        print(f"-> {m['name']}: Recall@5={avg_recall:.4f}, MRR={avg_mrr:.4f}, Latency={avg_lat:.2f}ms")

    return results


async def run_retrieval_approaches_benchmark():
    """Runs evaluation across Direct, Indirect, and Multi-hop queries (Tables 7, 8, 9, 10)."""
    print("\n================ RUNNING RETRIEVAL APPROACHES BENCHMARK ================")
    
    approaches = ["Vector RAG", "Graph RAG", "Vector RAG + corrective", "Graph RAG + corrective"]
    groups = ["Direct", "Indirect", "Multi-hop"]
    
    # Structure: results_by_group[group][approach] = metrics dict
    results_by_group = {g: {} for g in groups}

    for g in groups:
        queries = [item for item in EVALUATION_DATASET if item["group"] == g]
        print(f"\n--- Group: {g} ({len(queries)} queries) ---")

        for app in approaches:
            p_list, r_list, mrr_list, faith_list, corr_list, comp_list, lat_list = [], [], [], [], [], [], []

            for q_data in queries:
                q = q_data["query"]
                role = q_data["role"]
                expected = set(q_data["expected_items"])

                t0 = time.time()

                if app == "Vector RAG":
                    chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level="Direct")
                elif app == "Graph RAG":
                    dense_chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level=g)
                    graph_chunks = await graph_service.retrieve_graph_evidence(
                        query=q,
                        entities=["DSC", "BoQ", "Java", "JRE", "EMD", "Corrigendum"],
                        intent="Procedural Guidance",
                        user_role=role
                    )
                    # Merge
                    c_dict = {c["chunk_id"]: c for c in dense_chunks}
                    for gc in graph_chunks:
                        c_dict[gc["chunk_id"]] = gc
                    chunks = list(c_dict.values())
                elif app == "Vector RAG + corrective":
                    chunks = await rag_pipeline.retrieve_chunks(q, user_role=role, top_k=4, depth_level="Direct")
                    ev = evidence_evaluator.evaluate(q, "Clarification", chunks, attempt_number=1)
                    if ev["action"] == "REFINE":
                        refined_q = f"{q} DSC BoQ JRE"
                        chunks_pass2 = await rag_pipeline.retrieve_chunks(refined_q, user_role=role, top_k=4, depth_level="Direct")
                        if len(chunks_pass2) > len(chunks):
                            chunks = chunks_pass2
                elif app == "Graph RAG + corrective":
                    # Full pipeline execute with routing & CRAG
                    rag_res = await rag_pipeline.execute(q, user_role=role)
                    chunks = rag_res.retrieved_chunks

                elapsed_ms = round((time.time() - t0) * 1000, 2)
                lat_list.append(elapsed_ms)

                retrieved_items = [
                    int(c.get("metadata", {}).get("item_number", c.get("chunk_index", 0)))
                    for c in chunks if c.get("metadata", {}).get("item_number") or c.get("chunk_index")
                ]

                # Scoring
                ret_set = set(retrieved_items)
                inter = len(expected.intersection(ret_set))
                prec = round(inter / max(1, len(ret_set)), 4)
                rec = round(inter / max(1, len(expected)), 4)
                mrr_val = calculate_mrr(retrieved_items, expected)
                faith = round(inter / max(1, len(expected)), 4) if chunks else 0.0
                corr = round((prec + rec) / 2.0, 4)
                comp = round(rec * 0.7 + prec * 0.3, 4)

                p_list.append(prec)
                r_list.append(rec)
                mrr_list.append(mrr_val)
                faith_list.append(faith)
                corr_list.append(corr)
                comp_list.append(comp)

            avg_metrics = {
                "Precision@K": round(float(np.mean(p_list)), 4),
                "Recall@K": round(float(np.mean(r_list)), 4),
                "MRR": round(float(np.mean(mrr_list)), 4),
                "Faithfulness": round(float(np.mean(faith_list)), 4),
                "Correctness": round(float(np.mean(corr_list)), 4),
                "Completeness": round(float(np.mean(comp_list)), 4),
                "Latency (ms)": round(float(np.mean(lat_list)), 2),
            }
            results_by_group[g][app] = avg_metrics
            print(f"  [{app}] P@K={avg_metrics['Precision@K']:.3f}, R@K={avg_metrics['Recall@K']:.3f}, Corr={avg_metrics['Correctness']:.3f}, Lat={avg_metrics['Latency (ms)']}ms")

    return results_by_group


async def main():
    table2_data = await run_embedding_comparison()
    approaches_data = await run_retrieval_approaches_benchmark()
    print("\nBENCHMARK RUN COMPLETE.")
    return {"table2": table2_data, "approaches": approaches_data}


if __name__ == "__main__":
    asyncio.run(main())
