"""
Database Auto-Initializer and Comprehensive Seeder for ClarifiBids.
Checks if the local PostgreSQL database has tables, users, roles, knowledge graph,
and all official ingested document chunks.
If empty or missing documents, automatically creates schema, seeds roles & categories,
ingests all bundled official documents (PDFs & DOCX) with embeddings and role classification,
and populates the complete Knowledge Graph.
"""
import os
import sys
import logging
from pathlib import Path
from sqlalchemy import select, func

from app.core.database import async_engine, AsyncSessionLocal, Base
from app.models import Role, QueryCategory, DocumentChunk, KnowledgeSource, Document, DocumentEmbedding
from app.services.ingestion.smart_ingestion import ingest_document_file
from seed_graph import seed_knowledge_graph

logger = logging.getLogger("clarifibids.init_db")

BACKEND_DIR = Path(__file__).resolve().parent
DOCS_DIR = BACKEND_DIR / "data" / "documents"

ALL_OFFICIAL_DOCS = [
    ("FAQ.docx", "GePNIC FAQ Repository", "Official Frequently Asked Questions on GePNIC portal operations and guidelines."),
    ("ChecklistForBidders.pdf", "Bidder Checklist", "Technical checklists for handling support queries and issues of Bidders."),
    ("ChecklistForDepartmentUsers.pdf", "Department User Checklist", "Checklist for government department officials and procuring authorities."),
    ("eProcSupport-L1-SoP-ver-1.1-2024-July18-accepted.pdf", "Support L1 SoP", "Standard Operating Procedure for eProcurement support desk."),
    ("Bid submission with NDA.pdf", "Bid Submission NDA Manual", "Step-by-step procedure for bid submission with Non-Disclosure Agreement."),
    ("Bid submission_2 Stage Bidding.pdf", "2-Stage Bidding Manual", "Procedural manual for 2-stage tender bidding workflows."),
    ("Bidder user manual for QCBS flow.pdf", "QCBS Flow Manual", "Bidder user manual for Quality and Cost Based Selection bidding."),
    ("Bidder_Enrollment_Manual.pdf", "Bidder Enrollment Manual", "Step-by-step user manual for bidder registration and digital enrollment."),
]


async def ensure_database_initialized():
    """Ensure database schema, roles, all official document chunks, and knowledge graph exist."""
    logger.info("Verifying database schema, roles, and indexed chunks...")

    # 1. Ensure all schema tables exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Seed roles & categories & admin user if missing
    async with AsyncSessionLocal() as session:
        role_res = await session.execute(select(func.count(Role.id)))
        role_count = role_res.scalar() or 0
        if role_count == 0:
            logger.info("Seeding initial roles and query categories...")
            roles = [
                Role(name="Bidder", is_business_role=True, description="Standard Indian Domestic Bidder"),
                Role(name="Foreign Bidder", is_business_role=True, description="International Bidder Regulations"),
                Role(name="Department User", is_business_role=True, description="Procuring Entity Official"),
                Role(name="ADMIN", is_business_role=False, description="System Administration"),
            ]
            session.add_all(roles)

            categories = [
                QueryCategory(name="Portal Guidelines & Procurement Terms", description="General terms, standard IST time, basic features"),
                QueryCategory(name="Portal Usage & User Management", description="Bidder enrollment, login, registration, password recovery"),
                QueryCategory(name="Technical Assistance", description="DSC detection, JRE setup, browser, OS, driver configuration"),
                QueryCategory(name="Tender & Bid Information", description="Tender search, bid submission, BOQ, EMD fee, corrigenda, covers"),
                QueryCategory(name="Security Information", description="Digital signatures, encryption, CA validity, tamper prevention"),
                QueryCategory(name="Foreign Bidder Information", description="International vendor guidelines, foreign currency, foreign DSC"),
            ]
            session.add_all(categories)
            await session.commit()
            logger.info("Roles and categories seeded successfully.")

        # Ensure default ADMIN account exists
        from app.models import User, UserRole
        from app.core.auth import hash_password
        admin_check = await session.execute(select(User).where(User.username == "admin"))
        if not admin_check.scalars().first():
            role_q = await session.execute(select(Role).where(Role.name == "ADMIN"))
            admin_role = role_q.scalars().first()
            if admin_role:
                default_admin = User(
                    username="admin",
                    email="admin@gepnic.gov.in",
                    password_hash=hash_password("admin123"),
                    is_active=True
                )
                session.add(default_admin)
                await session.flush()
                session.add(UserRole(user_id=default_admin.id, role_id=admin_role.id))
                await session.commit()
                logger.info("Default admin user created (admin / admin123).")

        # 3. Check document chunks count
        count_res = await session.execute(select(func.count(DocumentChunk.id)))
        chunk_count = count_res.scalar() or 0

        if chunk_count >= 150:
            logger.info(f"Database is already indexed with {chunk_count} chunks. Skipping auto-ingestion.")
        else:
            logger.info(f"Database has only {chunk_count} chunks. Ingesting all bundled official documents...")
            for fn, sname, sdesc in ALL_OFFICIAL_DOCS:
                file_path = DOCS_DIR / fn
                if not file_path.is_file():
                    # Fallback check storage directory if present
                    storage_alt = BACKEND_DIR / "storage" / "documents" / fn
                    if storage_alt.is_file():
                        file_path = storage_alt

                if file_path.is_file():
                    logger.info(f"Auto-ingesting {fn} from {file_path}...")
                    file_bytes = file_path.read_bytes()
                    try:
                        res = await ingest_document_file(
                            file_name=fn,
                            file_bytes=file_bytes,
                            source_name=sname,
                            source_description=sdesc
                        )
                        logger.info(f"Ingested {fn}: {res.get('chunks_indexed', 0)} chunks.")
                    except Exception as e:
                        logger.error(f"Error ingesting {fn}: {e}")
                else:
                    logger.warning(f"Official document file {fn} not found on disk.")

    # 4. Ensure Knowledge Graph is seeded
    logger.info("Verifying knowledge graph schema...")
    try:
        await seed_knowledge_graph()
    except Exception as e:
        logger.warning(f"Knowledge graph check notice: {e}")
    logger.info("Database initialization and document ingestion check complete!")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ensure_database_initialized())
