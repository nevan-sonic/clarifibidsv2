# ClarifiBids v2: Grounded Technical Assistance & Procedural Intelligence Layer for GePNIC e-Procurement

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.x-61dafb.svg)](https://react.dev/)
[![PostgreSQL 18](https://img.shields.io/badge/PostgreSQL-18%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-BGE--Small--en--v1.5-orange.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ClarifiBids v2** is an enterprise AI technical assistance and procedural intelligence platform designed specifically for the **Government e-Procurement System of National Informatics Centre (GePNIC)** and the **Central Public Procurement Portal (CPPP) of India**.

It resolves high-friction tender submission blockers—such as Digital Signature Certificate (DSC) USB token detection errors, Java Runtime Environment (JRE) applet security restrictions, Bill of Quantities (BoQ) multi-currency formatting, EMD fee exemption validation, and multi-party bid opening committee decryption protocols—through **strict Role-Based Access Control (RBAC)**, an **Adaptive Multi-Strategy Retrieval Engine (Dynamic Vector RAG, Graph RAG, and Corrective RAG / CRAG)**, and **Official Document Page Citations**.

---

## 🏛 System Architecture & Component Diagram

The diagram below details the 5-tier architecture of ClarifiBids and the execution flow for user inquiries:

```mermaid
flowchart TD
    classDef client fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0f172a;
    classDef security fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#0f172a;
    classDef retrieval fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#0f172a;
    classDef crag fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#0f172a;
    classDef storage fill:#f1f5f9,stroke:#334155,stroke-width:2px,color:#0f172a;

    subgraph Client ["Client Layer (Frontend)"]
        UI["🖥️ React 18 GePNIC Portal\n(1-Click Role Login, Citation Badges, Graph Canvas)"]:::client
    end

    subgraph Gateway ["API & Security Gateway"]
        API["🛡️ FastAPI REST Gateway\n(JWT Bearer Auth + RBAC Guard)"]:::security
    end

    subgraph Classification ["Query Understanding Engine"]
        CL["🧠 Semantic Intent & Depth Classifier\n(BGE-Small Centroids + Entity Extractor)"]:::retrieval
    end

    subgraph Engine ["Adaptive Dynamic Retrieval Engine"]
        VR["🔍 Vector RAG\n(pgvector Cosine Similarity)"]:::retrieval
        GR["🕸️ Graph RAG\n(PostgreSQL Multi-Hop Traversal)"]:::retrieval
    end

    subgraph Evaluation ["Evaluation & Generation Layer"]
        CRAG{"⚖️ Corrective RAG (CRAG)\nRelevance >= 0.65\nSufficiency >= 0.70"}:::crag
        WEB["🌐 Web Search Fallback\n(Live MediaWiki REST API)"]:::crag
        LLM["🤖 Groq LPU Inference\n(Qwen 2.5 27B / Multi-Model Fallback)"]:::crag
    end

    subgraph Storage ["Persistent Knowledge Base"]
        DB[("🗄️ PostgreSQL 18 + pgvector\n(Dense Vectors + Ontology Graph + Audit Logs)")]:::storage
    end

    UI -->|"1. User Query & Role Token"| API
    API -->|"2. Check RBAC Permissions"| CL
    CL -->|"Direct Query"| VR
    CL -->|"Procedural / Multi-Hop"| GR
    VR <-->|"Dense Cosine Search"| DB
    GR <-->|"Multi-Hop Relational Traversal"| DB
    VR --> CRAG
    GR --> CRAG
    CRAG -->|"Sufficient"| LLM
    CRAG -.->|"Action: REFINE"| VR
    CRAG -->|"Action: OUT_OF_DOMAIN"| WEB --> LLM
    LLM -->|"Verified Answer + Exact Page Citations"| UI
    API -.->|"Asynchronous Audit Trail"| DB
```

---

## ⚡ Adaptive Dynamic Retrieval: Vector RAG, Graph RAG & CRAG

ClarifiBids dynamically selects the optimal retrieval paradigm based on linguistic complexity and entity requirements:

| Retrieval Strategy | When Selected | Mechanism | Example Inquiries |
| :--- | :--- | :--- | :--- |
| **Vector RAG** | `Direct` inquiries & term definitions | Cosine similarity over 384-dimensional dense embeddings (`BAAI/bge-small-en-v1.5`) via PostgreSQL `pgvector`. Fast and semantically granular. | *"What is the default date and time format in GePNIC?"*, *"What is the validity period of a DSC?"* |
| **Graph RAG** | `Procedural` & `Multi-source` / `Multi-hop` queries | Traverses PostgreSQL entity relationships (`ISSUES` $\rightarrow$ `TRIGGERS_PROCEDURE` $\rightarrow$ `REQUIRES_PREREQUISITE` $\rightarrow$ `GOVERNED_BY_CONDITION` $\rightarrow$ `SUPPORTS_EVIDENCE`) across 35 nodes and 33 edges. | *"What are the prerequisites and relationship steps for bid submission lifecycle and DSC token configuration?"*, *"Bid opener decryption failure and multi-party quorum"* |
| **Corrective RAG (CRAG)** | Quality verification across all retrievals | Evaluates candidate chunks across **Relevance** ($\ge 0.65$) and **Sufficiency Coverage** ($\ge 0.70$). Triggers entity query expansion if borderline, or invokes external web search fallback if absent. | Prevents hallucinations, refines technical queries, and safeguards compliance. |
| **Web Search Fallback** | Out-of-manual or non-procurement inquiries | Queries Wikipedia's MediaWiki and REST APIs in real time to provide verified summaries with direct source URLs. | *"What is post-quantum lattice cryptography?"* |
| **RBAC Guard** | Cross-role privilege escalation attempts | Intercepts unauthorized inquiries at the gateway level with `ACCESS_DENIED` and helpful role-guidance instructions. | Domestic bidder attempting to query internal department opener decryption or comparative charts. |

---

## 🛡️ Multi-Role Access Control (RBAC) Matrix

ClarifiBids enforces Role-Based Access Control at three independent architectural levels:

1. **API Gateway Level**: Restricted administration endpoints (`/api/v1/admin/*`) require a signed JWT token carrying `role: "ADMIN"`; unauthorized requests are rejected with `HTTP 403 Forbidden`.
2. **Pipeline Pre-Check Guard**: Inquiries attempting cross-role privilege escalation (e.g., domestic bidders asking for department-internal decryption keys or foreign currency policies) are intercepted immediately with `RBAC_GUARD`.
3. **Database Chunk-Level Filtering**: Document chunks stored in PostgreSQL are partitioned by `user_role` metadata (`GENERAL`, `Bidder`, `Foreign Bidder`, `Department User`). The retrieval query strictly filters out non-matching chunks.

| User Role | Target Audience | Permitted Operational Scope | Example Guarded Queries |
| :--- | :--- | :--- | :--- |
| **Bidder** | Domestic Indian Vendors | Domestic bidding workflows, Class-2/Class-3 DSC tokens, BoQ in INR, MSME/NSIC fee exemptions, JRE configuration. | Restricted from department opener decryption and foreign multi-currency BoQs. |
| **Foreign Bidder** | International Contractors | Overseas DSC procurement via Indian CAs, multi-currency BoQ submission rules, cross-border document verification. | Allowed foreign currency BoQ inquiries; restricted from internal department administration. |
| **Department User** | Procuring Entity Officials | Tender creation, comparative evaluation charts, 2-of-2 bid opener quorum, multi-party key decryption protocols. | Access to all administrative guidelines; isolated from unauthorized bidder modifications. |
| **ADMIN** | System Administrators | Central knowledge base management, live PDF/DOCX document upload, chunk classification, embedding generation. | Full platform oversight and management capabilities. |

---

## 📖 Official Document Citations & Inline PDF Viewer

ClarifiBids ensures that every procedural response is grounded with exact official references:
- **Exact Document & Page Citations**: Every answer provides citations with document name, item number, and page number (e.g. `[Source: FAQ.docx, Page 7, Item #25]`).
- **Interactive Page Preview**: Clicking the citation badge opens an extracted text modal displaying the exact clause and context from the document.
- **Direct PDF Viewer**: Clicking `Open Exact Page ↗` opens the official PDF directly in the browser navigated to the exact page (`#page=X`).

---

## 🏗 Technology Stack

| Tier | Technology | Specification | Function |
| :--- | :--- | :--- | :--- |
| **Frontend** | React 18 & TypeScript | React 19, Vite 8.3, Lucide Icons | GePNIC national portal design, 1-click role login, interactive knowledge graph modal. |
| **Styling** | Vanilla CSS | CSS Variables & Flex/Grid | High-fidelity government theme, accessible contrast, zero framework overhead. |
| **Backend API** | FastAPI | Python 3.12, Pydantic v2 | High-throughput asynchronous REST API, RBAC guards, audit trails. |
| **Database & Vectors** | PostgreSQL 18 + `pgvector` | PostgreSQL 18.x, pgvector 0.8+ | Relational data, 384-dimensional dense vectors, and audit history. |
| **Knowledge Graph** | Relational Graph Schema | SQLAlchemy (Asyncpg + Psycopg) | Ontological multi-hop network connecting issues, procedures, conditions, and rules. |
| **Embeddings** | SentenceTransformers | `BAAI/bge-small-en-v1.5` | Local CPU 384-dim dense embeddings (133 MB footprint, no API keys required). |
| **LLM Inference** | Groq LPU / local vLLM | `qwen/qwen3.8-27b` | Sub-second generative inference with automated multi-model fallback. |
| **Security** | PyJWT & Passlib | HMAC-SHA256 & Bcrypt | JWT session management and password hashing. |

---

## 🚀 Quickstart: Clone & Setup

ClarifiBids is engineered for **zero-touch setup**. On initial boot, FastAPI's `lifespan` automatically initializes tables, seeds roles & categories, ingests all 8 bundled official manuals, and builds the Knowledge Graph.

### Prerequisites
- **Python** (3.10 to 3.12)
- **Node.js** (v18 or higher) and `npm`
- **PostgreSQL 16+** with the **`pgvector`** extension
- **Groq API Key** (Free from [console.groq.com](https://console.groq.com/keys))

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/nevan-sonic/clarifibidsv2.git
cd clarifibidsv2
```

---

### Step 2: Database Initialization

1. Open `psql` or pgAdmin as your PostgreSQL superuser (`postgres`):
   ```sql
   CREATE DATABASE clarifibids_db;
   \c clarifibids_db
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   ```

---

### Step 3: Backend Setup & Auto-Ingestion

1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     # Windows:
     copy .env.example .env
     # Linux / macOS:
     cp .env.example .env
     ```
   - Edit `backend/.env` with your PostgreSQL password and Groq API key:
     ```env
     DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@127.0.0.1:5432/clarifibids_db
     SYNC_DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@127.0.0.1:5432/clarifibids_db
     GROQ_API_KEY=gsk_your_groq_api_key_here
     GROQ_MODEL=qwen/qwen3.8-27b
     ```

5. Launch the FastAPI server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

> 🌟 **Zero-Touch Auto-Ingestion**:
> When FastAPI boots up for the first time, `ensure_database_initialized()` automatically:
> - Creates all schema tables in PostgreSQL.
> - Seeds initial organizational roles (`Bidder`, `Foreign Bidder`, `Department User`, `ADMIN`) and query categories.
> - Parses, chunks, and generates 384-d dense embeddings for **all 8 bundled official documents** (`FAQ.docx`, `FAQ.pdf`, `ChecklistForBidders.pdf`, `ChecklistForDepartmentUsers.pdf`, `eProcSupport-L1-SoP...pdf`, `Bid submission with NDA.pdf`, `Bid submission_2 Stage Bidding.pdf`, `Bidder user manual for QCBS flow.pdf`, and `Bidder_Enrollment_Manual.pdf`).
> - Seeds and links the complete **35-node, 33-edge Knowledge Graph**.
> 
> **No manual ingestion scripts are needed!**

The backend API will be available at `http://127.0.0.1:8000`. Interactive Swagger documentation is at `http://127.0.0.1:8000/docs`.

---

### Step 4: Frontend Setup

1. Open a new terminal and navigate to `frontend`:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser at **`http://localhost:5173`**.

---

## 📊 Empirical Evaluation & Benchmark Results

The system was benchmarked across 30 labelled evaluation queries representing 6 distinct procurement domains across Direct, Indirect, and Multi-hop depth categories ($K=4$):

### 1. Retrieval Strategy Performance Comparison

| Query Group | Evaluated Approach | Precision@K | Recall@K | MRR | Faithfulness | Latency (ms) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Direct Queries** | **Vector RAG** | **0.150** | 0.200 | 0.200 | 0.200 | **119.29** | Fast direct clause lookup; optimal baseline |
| | Graph RAG | 0.125 | **0.350** | **0.258** | **0.350** | 131.80 | Broad contextual coverage |
| **Indirect Queries** | Vector RAG | 0.150 | 0.125 | 0.250 | 0.125 | **47.71** | Single-step retrieval |
| | **Graph RAG** | **0.150** | **0.258** | **0.300** | **0.258** | 121.72 | **+106% Recall** over Vector RAG |
| **Multi-hop Queries** | Vector RAG | 0.050 | 0.020 | 0.100 | 0.020 | **52.65** | Fails on disconnected conditions |
| | **Graph RAG** | **0.200** | **0.180** | **0.275** | **0.180** | 108.13 | **+800% Recall (9x gain)** via typed graph edges |

### 2. Embedding Model Comparison ($K=5$)

| Embedding Method | Type / Dimension | Recall@5 | MRR | Latency (ms) | Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **BAAI/bge-small-en-v1.5 (Selected)** | Dense, 384 | **0.2567** | 0.1878 | 33.95 | Highest semantic domain recall; local CPU execution |
| **all-MiniLM-L6-v2** | Dense, 384 | 0.1889 | 0.1972 | 18.88 | Fast CPU latency, lower procurement terminology recall |
| **BM25** | Sparse Lexical | 0.2039 | 0.1906 | **0.52** | Sub-millisecond exact token matching for error codes |
| **Hybrid (BM25 + BGE RRF)** | Hybrid Fusion | 0.2150 | **0.1978** | 31.50 | Highest overall MRR by combining dense vectors with exact tokens |

---

## 🔒 Security, Production Readiness & Enterprise Roadmap

1. **Air-Gapped & Sovereign Data Center Deployment**:
   - For sovereign government environments (e.g. NIC / CERT-In data centers), the cloud LLM can be replaced with an **on-premises open-source LLM** (such as **Qwen 2.5 14B/32B** or **Llama 3.3 70B**) running locally via **vLLM** or **Ollama**.
   - Simply update `LLM_PROVIDER=vllm` in `.env` with zero frontend or pipeline modifications.
2. **Keycloak / NIC Parichay SSO Integration**:
   - The FastAPI authentication layer supports standard OIDC/OAuth2 protocols for federation with government Single Sign-On (NIC SSO / Parichay).
3. **Dynamic Secrets Management**:
   - For enterprise Kubernetes deployments, secrets can be fetched dynamically from **HashiCorp Vault**, eliminating static secrets.

---

## 📄 License & Attribution

This project is developed under the **MIT License**.  
All official GePNIC manuals, guidelines, and FAQs are property of the **National Informatics Centre (NIC), Ministry of Electronics & Information Technology, Government of India**.
