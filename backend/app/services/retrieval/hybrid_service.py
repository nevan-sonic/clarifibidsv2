"""
Hybrid Retrieval Service (Dense + BM25 Lexical + Reciprocal Rank Fusion)
========================================================================
Supports:
1. Pure Dense Cosine Similarity (Vector RAG)
2. Pure Sparse Lexical Search (BM25)
3. Hybrid Search using Reciprocal Rank Fusion (RRF)
4. Query Instruction Prefix (BGE query formatting)
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models import DocumentChunk, DocumentEmbedding
from app.services.embedding.embedding_service import embedding_service


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", (text or "").lower())


class HybridRetrievalService:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_chunks: List[Dict[str, Any]] = []
        self._is_initialized = False

    async def initialize_corpus(self):
        """Preload chunks for BM25 search."""
        async with AsyncSessionLocal() as session:
            stmt = select(DocumentChunk)
            res = await session.execute(stmt)
            chunks = res.scalars().all()

            self._corpus_chunks = []
            tokenized_corpus = []

            for c in chunks:
                meta = c.metadata_ or {}
                chunk_data = {
                    "chunk_id": str(c.id),
                    "chunk_index": c.chunk_index,
                    "question": c.question or "",
                    "answer": c.answer or "",
                    "content": c.content or "",
                    "page_number": c.page_number or meta.get("page_number", ((c.chunk_index - 1) // 4) + 1),
                    "user_role": meta.get("user_role", "Bidder"),
                    "item_number": meta.get("item_number", c.chunk_index),
                    "metadata": meta
                }
                self._corpus_chunks.append(chunk_data)
                # Combine question + answer for lexical matching
                tokens = tokenize(f"{chunk_data['question']} {chunk_data['answer']}")
                tokenized_corpus.append(tokens)

            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            self._is_initialized = True

    async def ensure_initialized(self):
        if not self._is_initialized:
            await self.initialize_corpus()

    def bm25_search(
        self, 
        query: str, 
        user_role: str = "Bidder", 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform sparse lexical search via BM25."""
        if not self._bm25 or not self._corpus_chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        scored_pairs = []
        for idx, score in enumerate(scores):
            chunk = self._corpus_chunks[idx]
            chunk_role = chunk.get("user_role", "Bidder")
            is_general = chunk_role in ["GENERAL", "ALL", "Universal"]

            if not is_general and user_role != "ADMIN":
                if user_role == "Bidder" and chunk_role != "Bidder":
                    continue
                if user_role == "Foreign Bidder" and chunk_role != "Foreign Bidder":
                    continue
                if user_role == "Department User" and chunk_role != "Department User":
                    continue

            scored_pairs.append((float(score), chunk))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored_pairs[:top_k]:
            c_copy = dict(chunk)
            c_copy["bm25_score"] = round(score, 4)
            c_copy["similarity_score"] = round(min(1.0, score / 15.0), 4) # normalized indicator
            results.append(c_copy)
        return results

    async def hybrid_search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        user_role: str = "Bidder",
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine Dense Cosine Results and BM25 Sparse Results using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25)
        """
        await self.ensure_initialized()
        bm25_res = self.bm25_search(query, user_role=user_role, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, Dict[str, Any]] = {}

        # 1. Rank dense results
        for rank, chunk in enumerate(dense_results, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            chunks_by_id[cid] = chunk

        # 2. Rank BM25 results
        for rank, chunk in enumerate(bm25_res, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            if cid not in chunks_by_id:
                chunks_by_id[cid] = chunk

        # Sort by fused score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_chunks = []
        for cid in sorted_cids[:top_k]:
            chunk = chunks_by_id[cid]
            chunk_copy = dict(chunk)
            chunk_copy["rrf_score"] = round(rrf_scores[cid], 5)
            fused_chunks.append(chunk_copy)

        return fused_chunks


hybrid_retrieval_service = HybridRetrievalService()
