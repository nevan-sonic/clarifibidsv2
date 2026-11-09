"""
Smart Document Ingestion and Role Classifier
============================================
Parses PDF / DOCX / TXT with page tracking, semantic block chunking,
and LLM-driven RBAC role classification.
"""

import os
import io
import hashlib
import logging
from typing import List, Dict, Any, Optional
import docx
import pypdf
from groq import Groq

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models import KnowledgeSource, Document, DocumentChunk, DocumentEmbedding
from app.services.embedding.embedding_service import embedding_service
from sqlalchemy import select

logger = logging.getLogger("clarifibids.smart_ingestion")


def compute_file_checksum(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def classify_chunk_role_smart(text: str, file_name: str, groq_client: Optional[Groq] = None) -> str:
    """
    Classifies a chunk into:
      - 'Bidder': Domestic Indian bidder procedures, enrollment, BoQ upload, MSME EMD exemption
      - 'Foreign Bidder': International bidder regulations, overseas DSC, foreign currency BoQ
      - 'Department User': Internal procuring entity, tender creation, bid opener assignments, comparative charts
      - 'GENERAL': Common portal rules, IST time, browsers, login, passwords accessible to everyone
    """
    fn_lower = file_name.lower()
    text_lower = text.lower()

    # Document-level strong heuristics
    if "checklistfordepartmentusers" in fn_lower:
        return "Department User"
    if "checklistforbidders" in fn_lower or "bidder_enrollment" in fn_lower:
        if any(k in text_lower for k in ["foreign bidder", "foreign currency", "outside india"]):
            return "Foreign Bidder"
        return "Bidder"

    # Text keyword heuristics
    dept_keywords = [
        "tender creator", "bid opener", "nodal officer", "department user",
        "comparative chart", "financial opener", "technical opener",
        "create users", "reject bidder", "admin user", "eprocsupport-l1"
    ]
    if any(k in text_lower for k in dept_keywords):
        return "Department User"

    foreign_keywords = [
        "foreign bidder", "foreign consultant", "outside india", "out of india",
        "foreign currency", "international firm", "overseas dsc"
    ]
    if any(k in text_lower for k in foreign_keywords):
        return "Foreign Bidder"

    # Use LLM for intelligent classification
    if settings.GROQ_API_KEY:
        try:
            if groq_client is None:
                groq_client = Groq(api_key=settings.GROQ_API_KEY)

            prompt = (
                "You are an expert e-Procurement Access Control Classifier for the Indian Government GePNIC portal.\n"
                "Classify the following text chunk into strictly ONE audience role:\n"
                "- 'Bidder': Instructions, checklists, workflows specifically for domestic Indian bidders.\n"
                "- 'Foreign Bidder': Rules specifically for international/overseas bidders and foreign currencies.\n"
                "- 'Department User': Administrative guidelines, tender management, bid opener assignments, government officials.\n"
                "- 'GENERAL': Common portal terms, browser setup, IST date/time, general FAQs open to all users.\n\n"
                f"File: {file_name}\n"
                f"Text:\n\"\"\"{text[:800]}\"\"\"\n\n"
                "Respond with EXACTLY ONE word: Bidder, Foreign Bidder, Department User, or GENERAL."
            )
            resp = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            role_pred = (resp.choices[0].message.content or "").strip()
            for r in ["Department User", "Foreign Bidder", "Bidder", "GENERAL"]:
                if r.lower() in role_pred.lower():
                    return r
        except Exception as e:
            logger.warning(f"LLM classification fallback: {e}")

    return "GENERAL"


def extract_smart_chunks(file_name: str, file_bytes: bytes, max_words: int = 250) -> List[Dict[str, Any]]:
    """
    Extracts semantic chunks while strictly maintaining the page number and document title.
    """
    ext = os.path.splitext(file_name)[1].lower()
    raw_blocks = []

    if ext == ".docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        # Handle FAQ-style tables
        if doc.tables:
            for idx, table in enumerate(doc.tables, start=1):
                rows = [[c.text.strip() for c in r.cells] for r in table.rows]
                if len(rows) >= 2 and rows[0][0] and rows[1][0]:
                    q, a = rows[0][0], rows[1][0]
                    raw_blocks.append({
                        "question": q,
                        "answer": a,
                        "content": f"Question: {q}\nAnswer: {a}",
                        "page_number": ((idx - 1) // 4) + 1
                    })
        # Normal paragraphs
        p_text = []
        for p in doc.paragraphs:
            t = p.text.strip()
            if t:
                p_text.append(t)
                if len(" ".join(p_text).split()) >= 150:
                    raw_blocks.append({
                        "content": "\n".join(p_text),
                        "page_number": 1
                    })
                    p_text = []
        if p_text:
            raw_blocks.append({"content": "\n".join(p_text), "page_number": 1})

    elif ext == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            # Break page into coherent paragraphs
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            cur_para = []
            for p in paras:
                cur_para.append(p)
                if len(" ".join(cur_para).split()) >= 120:
                    raw_blocks.append({
                        "content": " ".join(cur_para),
                        "page_number": page_idx
                    })
                    cur_para = []
            if cur_para:
                raw_blocks.append({
                    "content": " ".join(cur_para),
                    "page_number": page_idx
                })

    elif ext in [".txt", ".md"]:
        text = file_bytes.decode("utf-8", errors="ignore")
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paras, start=1):
            raw_blocks.append({"content": p, "page_number": ((idx - 1) // 3) + 1})

    # Chunk sizing and normalization
    chunks = []
    chunk_idx = 1
    for block in raw_blocks:
        content = block["content"].strip()
        words = content.split()
        if not words or len(words) < 8:
            continue

        if len(words) <= max_words:
            chunks.append({
                "chunk_index": chunk_idx,
                "content": content,
                "question": block.get("question"),
                "answer": block.get("answer"),
                "page_number": block.get("page_number", 1),
                "token_count": len(words)
            })
            chunk_idx += 1
        else:
            for i in range(0, len(words), max_words):
                sub_words = words[i:i + max_words]
                sub_content = " ".join(sub_words)
                chunks.append({
                    "chunk_index": chunk_idx,
                    "content": sub_content,
                    "question": block.get("question"),
                    "answer": block.get("answer"),
                    "page_number": block.get("page_number", 1),
                    "token_count": len(sub_words)
                })
                chunk_idx += 1

    return chunks


async def ingest_single_document(
    file_name: str,
    file_bytes: bytes,
    source_name: Optional[str] = None,
    source_description: Optional[str] = None
) -> Dict[str, Any]:
    """Ingests a document, classifies each chunk, generates embeddings, and saves to DB."""
    checksum = compute_file_checksum(file_bytes)
    ext = os.path.splitext(file_name)[1].lower().replace(".", "").upper()
    s_name = source_name or f"Official Document: {file_name}"

    raw_chunks = extract_smart_chunks(file_name, file_bytes)
    if not raw_chunks:
        raise ValueError(f"No valid text could be extracted from {file_name}.")

    logger.info(f"Extracting & classifying {len(raw_chunks)} chunks for {file_name}...")

    # Classify roles for all chunks
    groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
    classified_chunks = []
    chunk_texts = []

    for c in raw_chunks:
        role = classify_chunk_role_smart(c["content"], file_name, groq_client)
        c["user_role"] = role
        c["metadata"] = {
            "source": s_name,
            "file_name": file_name,
            "page_number": c["page_number"],
            "user_role": role,
            "item_number": c["chunk_index"]
        }
        classified_chunks.append(c)
        chunk_texts.append(c["content"])

    # Compute dense embeddings in bulk
    embeddings = embedding_service.embed_batch(chunk_texts)

    # Persist in Database
    async with AsyncSessionLocal() as session:
        # Check if source exists
        res_src = await session.execute(select(KnowledgeSource).where(KnowledgeSource.source_name == s_name))
        source = res_src.scalars().first()
        if not source:
            source = KnowledgeSource(
                source_name=s_name,
                source_type="OFFICIAL_MANUAL",
                description=source_description or f"Official GePNIC Document: {file_name}",
                location=file_name
            )
            session.add(source)
            await session.flush()

        # Check if document already exists
        res_doc = await session.execute(select(Document).where(Document.file_name == file_name))
        doc = res_doc.scalars().first()
        if doc:
            # Delete old chunks
            del_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
            old_chunks = (await session.execute(del_stmt)).scalars().all()
            for oc in old_chunks:
                await session.delete(oc)
            doc.checksum = checksum
            doc.status = "RE_INDEXED"
        else:
            doc = Document(
                source_id=source.id,
                title=os.path.splitext(file_name)[0].replace("_", " "),
                document_type=ext,
                file_name=file_name,
                checksum=checksum,
                status="INDEXED"
            )
            session.add(doc)
            await session.flush()

        # Insert chunks and embeddings
        for idx, (c, emb) in enumerate(zip(classified_chunks, embeddings)):
            db_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=c["chunk_index"],
                content=c["content"],
                question=c.get("question"),
                answer=c.get("answer"),
                page_number=c["page_number"],
                token_count=c["token_count"],
                metadata_=c["metadata"]
            )
            session.add(db_chunk)
            await session.flush()

            db_emb = DocumentEmbedding(
                chunk_id=db_chunk.id,
                embedding=emb,
                embedding_model=settings.EMBEDDING_MODEL
            )
            session.add(db_emb)

        await session.commit()

    return {
        "file_name": file_name,
        "document_id": str(doc.id),
        "chunks_indexed": len(classified_chunks),
        "status": "SUCCESS"
    }


# Convenient alias
ingest_document_file = ingest_single_document
