"""
Ingest All GePNIC Source Documents Script
=========================================
Ingests the official files:
- FAQ.docx
- ChecklistForBidders.pdf
- ChecklistForDepartmentUsers.pdf
- eProcSupport-L1-SoP-ver-1.1-2024-July18-accepted.pdf
- Bid submission with NDA.pdf
- Bid submission_2 Stage Bidding.pdf
- Bidder user manual for QCBS flow.pdf
- Bidder_Enrollment_Manual.pdf
"""

import os
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest_all")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from app.services.ingestion.smart_ingestion import ingest_single_document
from seed_graph import seed_knowledge_graph

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
MANUAL_DIR = os.path.join(PARENT_DIR, "manual", "manual")

DOCUMENTS_TO_INGEST = [
    os.path.join(PARENT_DIR, "FAQ.docx"),
    os.path.join(PARENT_DIR, "ChecklistForBidders.pdf"),
    os.path.join(PARENT_DIR, "ChecklistForDepartmentUsers.pdf"),
    os.path.join(PARENT_DIR, "eProcSupport-L1-SoP-ver-1.1-2024-July18-accepted.pdf"),
    os.path.join(MANUAL_DIR, "Bid submission with NDA.pdf"),
    os.path.join(MANUAL_DIR, "Bid submission_2 Stage Bidding.pdf"),
    os.path.join(MANUAL_DIR, "Bidder user manual for QCBS flow.pdf"),
    os.path.join(MANUAL_DIR, "Bidder_Enrollment_Manual.pdf"),
]

async def main():
    total_chunks = 0
    for doc_path in DOCUMENTS_TO_INGEST:
        if not os.path.exists(doc_path):
            logger.warning(f"File not found: {doc_path}")
            continue

        file_name = os.path.basename(doc_path)
        logger.info(f"\n================ Ingesting: {file_name} ================")
        with open(doc_path, "rb") as f:
            file_bytes = f.read()

        res = await ingest_single_document(file_name, file_bytes)
        logger.info(f"Ingested {file_name}: {res['chunks_indexed']} chunks indexed.")
        total_chunks += res["chunks_indexed"]

    logger.info(f"\nAll documents successfully indexed! Total new chunks: {total_chunks}")
    
    # Re-seed knowledge graph so all newly indexed concepts are interconnected
    logger.info("Syncing Knowledge Graph nodes and edges with newly indexed evidence...")
    await seed_knowledge_graph()
    logger.info("Knowledge Graph sync complete!")

if __name__ == "__main__":
    asyncio.run(main())
