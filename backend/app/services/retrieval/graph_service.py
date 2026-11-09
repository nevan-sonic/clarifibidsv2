import asyncio
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models import GraphNode, GraphEdge, DocumentChunk

class GraphService:
    def __init__(self):
        pass

    async def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(GraphNode).where(GraphNode.node_type == node_type)
            res = await session.execute(stmt)
            nodes = res.scalars().all()
            return [
                {
                    "id": str(n.id),
                    "node_type": n.node_type,
                    "name": n.name,
                    "description": n.description,
                    "metadata": n.metadata_ or {}
                }
                for n in nodes
            ]

    async def traverse_prerequisites(self, concept_names: List[str], max_hops: int = 2) -> List[str]:
        """
        Traverse from given concept/issue names along REQUIRES_PREREQUISITE, TRIGGERS_PROCEDURE,
        and SUPPORTS_EVIDENCE edges to find connected document chunk IDs or item numbers.
        """
        async with AsyncSessionLocal() as session:
            # 1. Locate starting nodes
            start_stmt = select(GraphNode).where(
                GraphNode.name.in_(concept_names)
            )
            res = await session.execute(start_stmt)
            start_nodes = res.scalars().all()
            
            if not start_nodes:
                return []

            visited_node_ids = {n.id for n in start_nodes}
            current_node_ids = set(visited_node_ids)
            supporting_chunk_item_numbers = []

            for n in start_nodes:
                if n.node_type == "CHUNK" and n.metadata_.get("item_number"):
                    supporting_chunk_item_numbers.append(str(n.metadata_["item_number"]))

            # 2. Multi-hop traversal
            for _ in range(max_hops):
                if not current_node_ids:
                    break

                edge_stmt = select(GraphEdge, GraphNode).join(
                    GraphNode, GraphEdge.target_node_id == GraphNode.id
                ).where(GraphEdge.source_node_id.in_(current_node_ids))

                edge_res = await session.execute(edge_stmt)
                next_node_ids = set()

                for edge, target_node in edge_res.all():
                    if target_node.node_type == "CHUNK" and target_node.metadata_.get("item_number"):
                        supporting_chunk_item_numbers.append(str(target_node.metadata_["item_number"]))
                    
                    if target_node.id not in visited_node_ids:
                        visited_node_ids.add(target_node.id)
                        next_node_ids.add(target_node.id)

                current_node_ids = next_node_ids

            return list(dict.fromkeys(supporting_chunk_item_numbers))

    async def retrieve_graph_evidence(self, query: str, entities: List[str], intent: str, user_role: str = "Bidder") -> List[Dict[str, Any]]:
        """
        Graph RAG Retrieval: identifies matching graph nodes for the entities/intent,
        traverses connected procedure, condition, and chunk nodes, and calculates genuine
        cosine similarity against the user query.
        """
        concepts_to_query = []
        for ent in entities:
            concepts_to_query.append(ent)
        if intent:
            concepts_to_query.append(intent)

        if not concepts_to_query:
            return []

        async with AsyncSessionLocal() as session:
            # Look for nodes whose names match or contain concepts
            conditions = []
            for c in concepts_to_query:
                conditions.append(GraphNode.name.ilike(f"%{c}%"))

            from sqlalchemy import or_
            stmt = select(GraphNode).where(or_(*conditions))
            res = await session.execute(stmt)
            matched_nodes = res.scalars().all()
            
            if not matched_nodes:
                return []

            matched_names = [n.name for n in matched_nodes]
            chunk_item_numbers = await self.traverse_prerequisites(matched_names, max_hops=2)

            if not chunk_item_numbers:
                return []

            # Fetch the actual document chunks and embeddings corresponding to these item numbers
            from app.models import DocumentEmbedding
            from app.services.embedding.embedding_service import embedding_service
            import numpy as np

            chunks_stmt = (
                select(DocumentChunk, DocumentEmbedding.embedding)
                .join(DocumentEmbedding, DocumentChunk.id == DocumentEmbedding.chunk_id)
            )
            all_chunks_res = await session.execute(chunks_stmt)
            all_chunks = all_chunks_res.all()

            # Real query vector calculation
            query_vec = np.array(embedding_service.embed_text(query), dtype=np.float32)
            norm_q = np.linalg.norm(query_vec)
            if norm_q > 0:
                query_vec = query_vec / norm_q

            graph_chunks = []
            for ch, emb_list in all_chunks:
                meta = ch.metadata_ or {}
                item_num = str(meta.get("item_number", ch.chunk_index))
                chunk_role = meta.get("user_role", meta.get("applicable_role", "Bidder"))

                # Role enforcement on graph chunks: GENERAL accessible to all, role-specific restricted
                is_general = chunk_role in ["GENERAL", "ALL", "Universal"]
                if not is_general and chunk_role != user_role:
                    continue


                if item_num in chunk_item_numbers:
                    chunk_vec = np.array(emb_list, dtype=np.float32)
                    norm_c = np.linalg.norm(chunk_vec)
                    if norm_c > 0:
                        chunk_vec = chunk_vec / norm_c
                    real_sim = float(np.dot(query_vec, chunk_vec))

                    page_no = ch.page_number or meta.get("page_number", ((ch.chunk_index - 1) // 4) + 1)
                    graph_chunks.append({
                        "chunk_id": str(ch.id),
                        "chunk_index": ch.chunk_index,
                        "question": ch.question,
                        "answer": ch.answer,
                        "content": ch.content,
                        "page_number": page_no,
                        "similarity_score": round(real_sim, 4),
                        "metadata": meta,
                        "retrieval_source": "KNOWLEDGE_GRAPH"
                    })

            return graph_chunks

graph_service = GraphService()
