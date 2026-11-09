import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from sqlalchemy import select

logger = logging.getLogger("clarifibids.rag_pipeline")

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models import DocumentChunk, DocumentEmbedding
from app.services.embedding.embedding_service import embedding_service
from app.services.classification.query_understanding import query_understanding_service
from app.services.evaluation.evidence_evaluator import evidence_evaluator
from app.services.response.prompt_service import build_llm_messages
from app.llm.groq_provider import get_llm_provider
from app.services.retrieval.graph_service import graph_service
from app.services.retrieval.web_search_service import web_search_service


class RAGResult(BaseModel):
    query: str
    user_role: str
    classification: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    response_text: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    strategy: str = "VECTOR_RAG"

class RAGPipeline:
    def __init__(self):
        self.embedding_svc = embedding_service
        self.understanding_svc = query_understanding_service
        self.evaluator = evidence_evaluator
        self.web_search_svc = web_search_service
        
    async def retrieve_chunks(
        self, 
        query: str, 
        user_role: str = "Bidder", 
        top_k: int = 5,
        depth_level: str = "Direct"
    ) -> List[Dict[str, Any]]:
        # 1. Generate normalized 384-d semantic embedding for query
        query_vec = np.array(self.embedding_svc.embed_text(query), dtype=np.float32)
        norm_q = np.linalg.norm(query_vec)
        if norm_q > 0:
            query_vec = query_vec / norm_q

        # 2. Query PostgreSQL for document chunks and embeddings
        async with AsyncSessionLocal() as session:
            stmt = (
                select(DocumentChunk, DocumentEmbedding.embedding)
                .join(DocumentEmbedding, DocumentChunk.id == DocumentEmbedding.chunk_id)
            )
            res = await session.execute(stmt)
            rows = res.all()
            
            scored_chunks = []
            for chunk_row, emb_list in rows:
                meta = chunk_row.metadata_ or {}
                chunk_role = meta.get("user_role", "Bidder")
                
                # -------------------------------------------------------------
                # STRICT ROLE-BASED ACCESS CONTROL (RBAC) FILTER
                # -------------------------------------------------------------
                # GENERAL/Universal chunks: accessible to everyone.
                # Role-specific chunks: strictly restricted to matching role.
                # -------------------------------------------------------------
                is_general = chunk_role in ["GENERAL", "ALL", "Universal"]

                if not is_general:
                    if user_role == "Bidder" and chunk_role != "Bidder":
                        continue
                    if user_role == "Foreign Bidder" and chunk_role != "Foreign Bidder":
                        continue
                    if user_role == "Department User" and chunk_role != "Department User":
                        continue

                chunk_vec = np.array(emb_list, dtype=np.float32)
                norm_c = np.linalg.norm(chunk_vec)
                if norm_c > 0:
                    chunk_vec = chunk_vec / norm_c
                    
                similarity = float(np.dot(query_vec, chunk_vec))
                
                # Boost role-specific documents
                if chunk_role == user_role and user_role in ["Foreign Bidder", "Department User"]:
                    similarity = min(1.0, similarity * 1.15)
                    
                scored_chunks.append((similarity, chunk_row))
                
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            # Filter by relevance threshold
            valid_matches = [
                (sim, row) for sim, row in scored_chunks 
                if sim >= settings.RELEVANCE_THRESHOLD
            ]
            
            if not valid_matches:
                return []
                
            depth_max_k = {
                "Direct": 3,
                "Context-dependent": 4,
                "Procedural": 5,
                "Multi-source": 6
            }.get(depth_level, 4)
            
            top_matches = valid_matches[:depth_max_k]
            chunks = []
            for similarity, chunk_row in top_matches:
                meta = chunk_row.metadata_ or {}
                page_no = chunk_row.page_number or meta.get("page_number", 1)
                doc_name = meta.get("file_name", "GePNIC Manual")
                chunks.append({
                    "chunk_id": str(chunk_row.id),
                    "chunk_index": chunk_row.chunk_index,
                    "question": chunk_row.question,
                    "answer": chunk_row.answer,
                    "content": chunk_row.content,
                    "page_number": page_no,
                    "file_name": doc_name,
                    "similarity_score": round(similarity, 4),
                    "metadata": meta
                })
            return chunks

    async def execute(self, query: str, user_role: str = "Bidder") -> RAGResult:
        t0 = time.time()
        
        # 1. Dynamic Classification
        classification = self.understanding_svc.classify_query(query, user_role=user_role)
        intent = classification.get("intent", "General Inquiry")
        depth_level = classification.get("depth_level", "Direct")
        
        # --- RBAC PRE-CHECKS ---
        q_lower = query.lower()
        if user_role == "Bidder":
            dept_keywords = ["tender creator", "tender opener", "bid opener", "nodal officer", "auditor role", "create users", "comparative chart", "reject a bidder", "department user", "opening committee"]
            if any(k in q_lower for k in dept_keywords):
                elapsed_ms = round((time.time() - t0) * 1000, 2)
                return RAGResult(
                    query=query,
                    user_role=user_role,
                    classification=classification,
                    retrieved_chunks=[],
                    evaluation={
                        "is_relevant": False,
                        "is_sufficient": False,
                        "coverage_score": 0.0,
                        "action": "ACCESS_DENIED",
                        "reason": "Access Restricted: Domestic Bidders are strictly prohibited from viewing internal departmental procurement workflows."
                    },
                    response_text="**Access Restricted (RBAC Policy)**:\n\nAs a **Domestic Bidder**, you do not have authorization to access internal **Department User / Procuring Authority** administrative procedures (such as tender creation, opening quorum configuration, or comparative evaluation charts).\n\nTo view departmental procedures, please authenticate with a **Department User** account or contact the designated Nodal Officer.",
                    sources=[],
                    latency_ms=elapsed_ms,
                    strategy="RBAC_GUARD"
                )
            if any(k in q_lower for k in ["foreign bidder", "foreign consultant", "outside india", "out of india", "currencies other than indian rupees", "overseas dsc"]):
                elapsed_ms = round((time.time() - t0) * 1000, 2)
                return RAGResult(
                    query=query,
                    user_role=user_role,
                    classification=classification,
                    retrieved_chunks=[],
                    evaluation={
                        "is_relevant": False,
                        "is_sufficient": False,
                        "coverage_score": 0.0,
                        "action": "ACCESS_DENIED",
                        "reason": "Access Restricted: International vendor policies and foreign currency rules are restricted to the Foreign Bidder role."
                    },
                    response_text="**Access Restricted (RBAC Policy)**:\n\nThis inquiry pertains specifically to **Foreign Bidder / International Vendor Regulations** (including overseas DSC issuance and multi-currency BoQ submissions).\n\nAs a domestic **Bidder**, standard Indian procurement rules (INR currency, domestic Certifying Authorities) apply to your account. To view international vendor guidelines, please switch your role to **Foreign Bidder**.",
                    sources=[],
                    latency_ms=elapsed_ms,
                    strategy="RBAC_GUARD"
                )

        # 2. Candidate Retrieval from Indexed Manuals
        chunks = await self.retrieve_chunks(query, user_role=user_role, top_k=settings.TOP_K, depth_level=depth_level)
        
        # Determine Strategy: Vector RAG vs. Graph RAG
        strategy = "GRAPH_RAG" if depth_level in ["Procedural", "Multi-source"] else "VECTOR_RAG"

        # If Graph RAG is selected, augment with relational knowledge graph traversal
        if strategy == "GRAPH_RAG":
            graph_chunks = await graph_service.retrieve_graph_evidence(
                query=query,
                entities=classification.get("entities", []),
                intent=intent,
                user_role=user_role
            )
            if graph_chunks:
                existing_ids = {c["chunk_id"] for c in chunks}
                for gc in graph_chunks:
                    if gc["chunk_id"] not in existing_ids:
                        chunks.append(gc)
                        existing_ids.add(gc["chunk_id"])

        # 3. Corrective RAG (CRAG) Sufficiency Evaluation
        eval_result = self.evaluator.evaluate(query, intent, chunks, attempt_number=1)
        
        # If chunks are insufficient, attempt Query Expansion (Corrective Pass)
        if eval_result["action"] == "REFINE" and len(classification.get("entities", [])) > 0:
            strategy += " + CORRECTIVE"
            refined_query = f"{query} {' '.join(classification.get('entities', []))}"
            chunks_pass2 = await self.retrieve_chunks(refined_query, user_role=user_role, top_k=settings.TOP_K, depth_level=depth_level)
            eval_pass2 = self.evaluator.evaluate(refined_query, intent, chunks_pass2, attempt_number=2)
            if eval_pass2["is_sufficient"] or len(chunks_pass2) > len(chunks):
                chunks = chunks_pass2
                eval_result = eval_pass2

        # 4. Out-of-Domain or Insufficient Knowledge -> FALLBACK TO WEB SEARCH
        top_score = chunks[0].get("similarity_score", 0.0) if chunks else 0.0
        is_weak_evidence = top_score < settings.RELEVANCE_THRESHOLD
        
        if not chunks or is_weak_evidence or not eval_result.get("is_sufficient", True) or eval_result.get("action") in ["OUT_OF_DOMAIN", "IRRELEVANT_QUERY", "REFINE", "EXHAUSTED"]:
            logger.info(f"Query '{query}' weak/insufficient in official manuals. Querying web search...")
            web_evidence = self.web_search_svc.search(query, max_results=3)
            if web_evidence:
                chunks = web_evidence
                strategy = "WEB_SEARCH_FALLBACK"
                eval_result = {
                    "is_relevant": True,
                    "is_sufficient": True,
                    "coverage_score": 0.85,
                    "action": "PROCEED_WEB",
                    "reason": "Retrieved verified web evidence for inquiry outside indexed manuals."
                }

        # 6. Response Synthesis via Groq LLM
        if not chunks:
            response_text = "The official GePNIC knowledge base and verified external sources do not contain sufficient evidence to answer this inquiry. ClarifiBids strictly provides grounded answers with official citations."
            sources = []
        else:
            messages = build_llm_messages(query, user_role, chunks)
            llm = get_llm_provider()
            try:
                llm_resp = await llm.generate(messages, temperature=0.1, max_tokens=settings.MAX_OUTPUT_TOKENS)
                response_text = llm_resp.content
            except Exception as e:
                err_detail = str(e).strip() or type(e).__name__
                logger.error(f"LLM generation failed: {err_detail}", exc_info=True)
                # Resilient fallback: ground directly on retrieved official evidence
                top_ans = chunks[0].get("answer") or chunks[0].get("content") or ""
                doc_name = chunks[0].get("metadata", {}).get("file_name", "GePNIC Official Manual")
                page_no = chunks[0].get("page_number", 1)
                item_no = chunks[0].get("metadata", {}).get("item_number", chunks[0].get("chunk_index", 1))
                if top_ans:
                    response_text = (
                        f"**Procedural Guidance (Official Grounding)**:\n\n{top_ans.strip()}\n\n"
                        f"---\n**Document Citations**\n- [Source: {doc_name}, Page {page_no}, Item #{item_no}]\n\n"
                        f"*(Direct ground fallback due to transient upstream AI connection latency: {err_detail})*"
                    )
                else:
                    response_text = f"Temporary service notice: The procedural intelligence engine encountered connection latency ({err_detail}). Please retry your inquiry."
                
            sources = []
            for c in chunks:
                meta = c.get("metadata", {})
                doc_name = meta.get("file_name", meta.get("source", "GePNIC Documentation"))
                page_no = c.get("page_number", meta.get("page_number", 1))
                sources.append({
                    "source": doc_name,
                    "file_name": doc_name,
                    "page_number": page_no,
                    "item_number": meta.get("item_number", c.get("chunk_index", 1)),
                    "similarity_score": c.get("similarity_score", 0.8),
                    "is_web": meta.get("is_web", False),
                    "url": meta.get("url", ""),
                    "content_snippet": (c.get("answer") or c.get("content") or "")[:250]
                })
            
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        
        return RAGResult(
            query=query,
            user_role=user_role,
            classification=classification,
            retrieved_chunks=chunks,
            evaluation=eval_result,
            response_text=response_text,
            sources=sources,
            latency_ms=elapsed_ms,
            strategy=strategy
        )

rag_pipeline = RAGPipeline()
