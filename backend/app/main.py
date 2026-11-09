import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query as FastQuery
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, func

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.auth import (
    hash_password, verify_password, create_access_token, get_current_user_optional, require_admin_user
)
from app.models import (
    User, Role, UserRole, Query, QueryClassification, QueryCategory,
    RetrievalRun, RetrievedEvidence, EvidenceEvaluation, Response,
    Document, DocumentChunk, EvaluationResult, EvaluationQuery
)
from app.services.retrieval.rag_pipeline import rag_pipeline, RAGResult
from app.services.classification.query_understanding import CATEGORY_DESCRIPTIONS
from app.services.evaluation.benchmark_service import benchmark_service
from app.services.ingestion.smart_ingestion import ingest_document_file
from setup_db import ensure_database_initialized

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await ensure_database_initialized()
    except Exception as e:
        logger.error(f"Database auto-initialization error: {e}")
    yield

app = FastAPI(
    title="ClarifiBids API",
    description="Intelligent Assistance Layer over GePNIC eProcurement Portal",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------

class ChatQueryRequest(BaseModel):
    query_text: str = Field(..., min_length=2, max_length=500)
    session_id: str = "default_session"
    user_role: str = Field("Bidder", pattern="^(Bidder|Foreign Bidder|Department User)$")

class ChatQueryResponse(BaseModel):
    query: str
    user_role: str
    classification: Dict[str, Any]
    answer: str
    sources: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    latency_ms: float
    strategy: str = "VECTOR_RAG"

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=4)
    role: str = Field("Bidder", pattern="^(Bidder|Foreign Bidder|Department User|ADMIN)$")

class LoginRequest(BaseModel):
    role: Optional[str] = "Bidder"
    username: Optional[str] = None
    password: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

# ---------------------------------------------------------
# Core Health & Taxonomy Routes
# ---------------------------------------------------------

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "app_env": settings.APP_ENV,
        "database": "PostgreSQL 18 + pgvector",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.get("/api/v1/taxonomy/categories")
def get_categories():
    return [
        {"name": name, "description": desc}
        for name, desc in CATEGORY_DESCRIPTIONS.items()
    ]

# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------

@app.post("/api/v1/auth/register", response_model=AuthResponse)
async def register_user(req: RegisterRequest):
    async with AsyncSessionLocal() as session:
        stmt = select(User).where(User.username == req.username)
        res = await session.execute(stmt)
        if res.scalars().first():
            raise HTTPException(status_code=400, detail="Username already registered")

        role_stmt = select(Role).where(Role.name == req.role)
        role_res = await session.execute(role_stmt)
        role_obj = role_res.scalars().first()
        if not role_obj:
            raise HTTPException(status_code=400, detail=f"Role '{req.role}' not found")

        new_user = User(
            username=req.username,
            email=req.email,
            password_hash=hash_password(req.password),
            is_active=True
        )
        session.add(new_user)
        await session.flush()

        user_role = UserRole(user_id=new_user.id, role_id=role_obj.id)
        session.add(user_role)
        await session.commit()

        token = create_access_token(new_user.id, new_user.username, req.role)
        return AuthResponse(
            access_token=token,
            user={"id": str(new_user.id), "username": new_user.username, "email": new_user.email, "role": req.role}
        )

@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login_user(req: LoginRequest):
    async with AsyncSessionLocal() as session:
        target_role = req.role or "Bidder"
        
        # If credentials provided, authenticate existing user
        if req.username and req.password:
            stmt = select(User).where(User.username == req.username)
            res = await session.execute(stmt)
            user = res.scalars().first()
            if not user or not verify_password(req.password, user.password_hash or ""):
                raise HTTPException(status_code=401, detail="Invalid username or password")

            ur_stmt = select(Role.name).join(UserRole, Role.id == UserRole.role_id).where(UserRole.user_id == user.id)
            ur_res = await session.execute(ur_stmt)
            role_name = ur_res.scalar_one_or_none() or "Bidder"

            token = create_access_token(user.id, user.username, role_name)
            return AuthResponse(
                access_token=token,
                user={"id": str(user.id), "username": user.username, "email": user.email, "role": role_name}
            )

        # 1-Click Fast Login by Role (no username/password required)
        role_stmt = select(Role).where(Role.name == target_role)
        role_res = await session.execute(role_stmt)
        role_obj = role_res.scalars().first()
        if not role_obj:
            raise HTTPException(status_code=400, detail=f"Role '{target_role}' not found in system")

        # Find or create a standard user for this role
        clean_name = target_role.lower().replace(" ", "_")
        default_username = f"user_{clean_name}"
        user_stmt = select(User).where(User.username == default_username)
        user_res = await session.execute(user_stmt)
        user = user_res.scalars().first()

        if not user:
            user = User(
                username=default_username,
                email=f"{clean_name}@eprocure.gov.in",
                password_hash=hash_password("clarifibids2026"),
                is_active=True
            )
            session.add(user)
            await session.flush()
            session.add(UserRole(user_id=user.id, role_id=role_obj.id))
            await session.commit()
        else:
            # Ensure user has this role mapped
            check_ur = await session.execute(
                select(UserRole).where(UserRole.user_id == user.id, UserRole.role_id == role_obj.id)
            )
            if not check_ur.scalars().first():
                session.add(UserRole(user_id=user.id, role_id=role_obj.id))
                await session.commit()

        token = create_access_token(user.id, user.username, target_role)
        return AuthResponse(
            access_token=token,
            user={
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": target_role
            }
        )

@app.get("/api/v1/auth/me")
async def get_me(user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ---------------------------------------------------------
# Chat & RAG Retrieval
# ---------------------------------------------------------

@app.post("/api/v1/chat/query", response_model=ChatQueryResponse)
async def handle_chat_query(req: ChatQueryRequest):
    try:
        rag_res: RAGResult = await rag_pipeline.execute(
            query=req.query_text,
            user_role=req.user_role
        )

        # Audit persistence (non-blocking failure tolerance)
        try:
            async with AsyncSessionLocal() as session:
                db_query = Query(
                    session_id=req.session_id,
                    query_text=rag_res.query,
                    user_role=rag_res.user_role
                )
                session.add(db_query)
                await session.flush()

                cat_name = rag_res.classification.get("category", "")
                cat_stmt = select(QueryCategory.id).where(QueryCategory.name == cat_name)
                cat_res = await session.execute(cat_stmt)
                cat_id = cat_res.scalar_one_or_none()

                db_class = QueryClassification(
                    query_id=db_query.id,
                    category_id=cat_id,
                    depth_level=rag_res.classification.get("depth_level", "Direct"),
                    confidence_score=rag_res.classification.get("confidence_score", 0.0),
                    entities=rag_res.classification.get("entities", []),
                    required_context=rag_res.classification.get("required_context", {})
                )
                session.add(db_class)

                db_run = RetrievalRun(
                    query_id=db_query.id,
                    strategy=rag_res.strategy,
                    attempt_number=1,
                    top_k=len(rag_res.retrieved_chunks),
                    latency_ms=rag_res.latency_ms,
                    status="COMPLETED"
                )
                session.add(db_run)
                await session.flush()

                for idx, c in enumerate(rag_res.retrieved_chunks, 1):
                    c_id_str = c.get("chunk_id")
                    if c_id_str:
                        db_ev = RetrievedEvidence(
                            retrieval_run_id=db_run.id,
                            chunk_id=uuid.UUID(c_id_str),
                            similarity_score=c.get("similarity_score", 0.0),
                            rank=idx,
                            is_selected=True
                        )
                        session.add(db_ev)

                db_eval = EvidenceEvaluation(
                    retrieval_run_id=db_run.id,
                    is_relevant=rag_res.evaluation.get("is_relevant", False),
                    is_sufficient=rag_res.evaluation.get("is_sufficient", False),
                    coverage_score=rag_res.evaluation.get("coverage_score", 0.0),
                    action_taken=rag_res.evaluation.get("action", "PROCEED"),
                    reason=rag_res.evaluation.get("reason", "")
                )
                session.add(db_eval)

                db_resp = Response(
                    query_id=db_query.id,
                    retrieval_run_id=db_run.id,
                    response_text=rag_res.response_text,
                    model_name=settings.GROQ_MODEL,
                    provider=settings.LLM_PROVIDER,
                    prompt_version="v1",
                    groundedness_score=rag_res.evaluation.get("coverage_score", 0.0),
                    latency_ms=rag_res.latency_ms
                )
                session.add(db_resp)
                await session.commit()
        except Exception as audit_err:
            logger.error(f"Audit log persistence error: {audit_err}")

        return ChatQueryResponse(
            query=rag_res.query,
            user_role=rag_res.user_role,
            classification=rag_res.classification,
            answer=rag_res.response_text,
            sources=rag_res.sources,
            evaluation=rag_res.evaluation,
            latency_ms=rag_res.latency_ms,
            strategy=rag_res.strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Admin Document Management & Ingestion (RBAC Protected)
# ---------------------------------------------------------

@app.post("/api/v1/admin/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    # Strict RBAC guard: Verifies JWT token has role 'ADMIN'; rejects others with 403 Forbidden
    current_admin: Dict[str, Any] = Depends(require_admin_user)
):
    """
    Admin-only endpoint for uploading knowledge sources.
    
    Workflow:
    1. Reads the raw binary file bytes (DOCX, PDF, or TXT).
    2. Chunks the document into logical paragraphs/Q&As.
    3. Prompts Groq AI with zero-shot classification to tag each chunk with an access role:
       - 'GENERAL': Accessible to all users
       - 'Bidder': Domestic Indian bidders
       - 'Foreign Bidder': International bidders
       - 'Department User': Internal government tender officials
    4. Computes 384-dimensional dense semantic embeddings using BAAI/bge-small-en-v1.5.
    5. Saves document records, chunks, and embeddings into PostgreSQL.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        s_name = source_name or file.filename or "Uploaded Document"
        s_desc = description or f"Uploaded via admin console: {file.filename}"

        # Persist raw uploaded file into storage directory for preview and download
        storage_dir = Path(__file__).resolve().parent.parent / "storage" / "documents"
        storage_dir.mkdir(parents=True, exist_ok=True)
        if file.filename:
            target_file_path = storage_dir / Path(file.filename).name
            target_file_path.write_bytes(content)

        result = await ingest_document_file(
            file_name=file.filename or "uploaded_doc.txt",
            file_bytes=content,
            source_name=s_name,
            source_description=s_desc
        )
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {file.filename}",
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/api/v1/admin/documents")
async def list_documents(current_admin: Dict[str, Any] = Depends(require_admin_user)):
    """Admin-only endpoint to view registered documents and chunk statistics."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(
                Document.id,
                Document.title,
                Document.file_name,
                Document.document_type,
                Document.status,
                Document.created_at,
                func.count(DocumentChunk.id).label("chunk_count")
            )
            .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
            .group_by(Document.id)
            .order_by(Document.created_at.desc())
        )
        res = await session.execute(stmt)
        docs = [
            {
                "id": str(row.id),
                "title": row.title,
                "file_name": row.file_name,
                "document_type": row.document_type,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "chunk_count": row.chunk_count
            }
            for row in res.all()
        ]
        return {"documents": docs}

# ---------------------------------------------------------
# Evaluation Benchmark Routes
# ---------------------------------------------------------

@app.post("/api/v1/evaluation/run-benchmark")
async def trigger_benchmark():
    try:
        report = await benchmark_service.run_benchmark()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/evaluation/results")
async def get_evaluation_history():
    async with AsyncSessionLocal() as session:
        stmt = select(EvaluationResult, EvaluationQuery).join(
            EvaluationQuery, EvaluationResult.evaluation_query_id == EvaluationQuery.id
        ).order_by(EvaluationResult.created_at.desc()).limit(20)
        res = await session.execute(stmt)
        history = [
            {
                "query": q.query_text,
                "depth_level": q.depth_level,
                "approach": r.approach,
                "precision_at_k": r.precision_at_k,
                "recall_at_k": r.recall_at_k,
                "faithfulness": r.faithfulness,
                "context_completeness": r.context_completeness,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r, q in res.all()
        ]
        return {"total": len(history), "results": history}

# ---------------------------------------------------------
# Document Raw Download & Page Interactive Preview Routes
# ---------------------------------------------------------

DOC_SEARCH_DIRS = [
    Path(__file__).resolve().parent.parent / "data" / "documents",
    Path(__file__).resolve().parent.parent / "storage" / "documents",
]

def find_document_file(file_name: str) -> Optional[Path]:
    clean_name = Path(file_name).name
    for search_dir in DOC_SEARCH_DIRS:
        cand = search_dir / clean_name
        if cand.is_file():
            return cand
    return None

@app.get("/api/v1/graph/visualize")
async def get_knowledge_graph_data():
    async with AsyncSessionLocal() as session:
        from app.models import GraphNode, GraphEdge
        nodes_res = await session.execute(select(GraphNode))
        nodes = [
            {
                "id": str(n.id),
                "name": n.name,
                "node_type": n.node_type,
                "description": n.description
            }
            for n in nodes_res.scalars().all()
        ]

        edges_res = await session.execute(select(GraphEdge))
        edges = [
            {
                "id": str(e.id),
                "source": str(e.source_node_id),
                "target": str(e.target_node_id),
                "relationship_type": e.relationship_type,
                "weight": e.weight
            }
            for e in edges_res.scalars().all()
        ]

        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges
        }

@app.get("/api/v1/documents/view/{file_name}")
async def view_document_pdf(
    file_name: str,
    page: Optional[int] = FastQuery(1, description="Page number to open")
):
    clean_name = Path(file_name).name
    target_name = clean_name
    if clean_name.lower().endswith(".docx"):
        pdf_alt = clean_name[:-5] + ".pdf"
        if find_document_file(pdf_alt):
            target_name = pdf_alt

    file_path = find_document_file(target_name)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found.")

    media_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "application/octet-stream"
    headers = {
        "Content-Disposition": f'inline; filename="{file_path.name}"'
    }
    return FileResponse(path=str(file_path), media_type=media_type, headers=headers)

@app.get("/api/v1/documents/preview")
async def get_document_page_preview(
    file_name: str = FastQuery(..., description="Document file name e.g. FAQ.docx or ChecklistForBidders.pdf"),
    page_number: Optional[int] = FastQuery(1, description="Target page number (1-indexed)"),
    item_number: Optional[int] = FastQuery(None, description="Optional FAQ Item or Clause number")
):
    file_path = find_document_file(file_name)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found.")

    suffix = file_path.suffix.lower()
    total_pages = 1
    page_text = ""
    highlight_clause = ""

    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            target_idx = max(0, min(page_number - 1, total_pages - 1))
            page_text = reader.pages[target_idx].extract_text() or "(No readable text on this page)"
            actual_page = target_idx + 1
        elif suffix in [".docx", ".doc"]:
            from docx import Document as DocxDocument
            doc = DocxDocument(str(file_path))
            rows_data = []
            # In FAQ.docx, each table contains 2 rows: row 0 is Question, row 1 is Answer
            for table in doc.tables:
                if len(table.rows) >= 2:
                    q = table.rows[0].cells[0].text.strip()
                    a = table.rows[1].cells[0].text.strip()
                    # Extract starting item number if present (e.g. "2. What are...")
                    prefix_num = None
                    tokens = q.split(".", 1)
                    if tokens[0].strip().isdigit():
                        prefix_num = int(tokens[0].strip())
                    rows_data.append((prefix_num or (len(rows_data) + 1), q, a))

            total_pages = max(1, len(rows_data))
            actual_page = page_number or 1

            if item_number is not None:
                matched = [r for r in rows_data if r[0] == item_number]
                if matched:
                    num, q, a = matched[0]
                    highlight_clause = q
                    page_text = f"ITEM #{num}\nQUESTION:\n{q}\n\nOFFICIAL CLAUSE / PROCEDURE:\n{a}"
                    actual_page = num

            if not page_text:
                if 1 <= actual_page <= len(rows_data):
                    num, q, a = rows_data[actual_page - 1]
                    highlight_clause = q
                    page_text = f"ITEM #{num}\nQUESTION:\n{q}\n\nOFFICIAL CLAUSE / PROCEDURE:\n{a}"
                else:
                    all_text = [p.text for p in doc.paragraphs if p.text.strip()]
                    page_text = "\n\n".join(all_text[:15]) or "No content extracted"
        else:
            actual_page = 1
            page_text = file_path.read_text(errors="ignore")[:3000]

        return {
            "file_name": file_path.name,
            "page_number": actual_page,
            "total_pages": total_pages,
            "highlight_clause": highlight_clause,
            "page_text": page_text,
            "download_url": f"http://127.0.0.1:8000/api/v1/documents/raw/{file_path.name}"
        }
    except Exception as e:
        logger.exception("Error extracting page preview")
        raise HTTPException(status_code=500, detail=f"Failed to extract document preview: {str(e)}")

