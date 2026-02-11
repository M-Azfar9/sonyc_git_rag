# 🤖 SONYC — GitHub Repo AI Agent

> **Module 2: GitHub Repo AI Agent** — An AI agent that ingests GitHub repositories, understands everything inside them (code, issues, branches, commit history, and more), and lets users query them intelligently through a conversational RAG interface.

---

## 📋 Problem Statement

> *Build an AI agent that can ingest a GitHub repo (public or private), understand everything inside it — code, issues, branches, commit history, and more — and let users query it intelligently.*

**Required Pipeline**: `Repo context → vectorized → searchable → report generation → conversational`

| # | Requirement | Description |
|---|---|---|
| R1 | **Real Time** | Auto-detect every push on the repo and keep context fresh |
| R2 | **Multiple Projects** | Support more than one repo or workspace |
| R3 | **Persistent Sessions** | Users can return and continue where they left off |
| R4 | **Usable UI** | Clear, functional interface for querying and exploring |

---

## ✅ Core Features — Requirement Mapping

Every requirement from the problem statement is fully implemented in our backend. Here's the exact mapping:

### 🔗 Pipeline: `Repo Context → Vectorized → Searchable → Report Generation → Conversational`

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           CORE PIPELINE (FULLY IMPLEMENTED)                         │
│                                                                                      │
│  1. REPO CONTEXT           2. VECTORIZED          3. SEARCHABLE                      │
│  ┌──────────────────┐      ┌────────────────┐     ┌──────────────────┐               │
│  │ github_service.py│      │ ChromaDB +     │     │ MMR Retriever    │               │
│  │                  │─────▶│ MistralAI      │────▶│ (k=5 vectors)   │               │
│  │ • Source Code    │      │ Embeddings     │     │ Semantic Search  │               │
│  │ • Issues         │      │ ("mistral-     │     │ on vectorized    │               │
│  │ • PRs            │      │  embed")       │     │ repo context     │               │
│  │ • Commits        │      │                │     │                  │               │
│  │ • Branches       │      │ Dynamic chunk  │     └────────┬─────────┘               │
│  └──────────────────┘      │ sizing         │              │                         │
│                            └────────────────┘              │                         │
│                                                            ▼                         │
│  4. REPORT GENERATION                        5. CONVERSATIONAL                       │
│  ┌──────────────────┐                        ┌──────────────────┐                    │
│  │ /git_report      │                        │ /chat/stream     │                    │
│  │                  │                        │                  │                    │
│  │ 4 report types:  │                        │ RAG-grounded     │                    │
│  │ • Full           │                        │ streaming chat   │                    │
│  │ • Architecture   │                        │ with repo        │                    │
│  │ • Dependencies   │                        │ context          │                    │
│  │ • Code Quality   │                        │                  │                    │
│  └──────────────────┘                        └──────────────────┘                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

### R1: Real Time — Auto-detect pushes and keep context fresh ✅

**Implementation**: GitHub Webhook system with HMAC-SHA256 signature verification

| Component | File | How It Works |
|---|---|---|
| Webhook Endpoint | `main.py` → `POST /webhook/github` | Receives GitHub webhook events for `push`, `issues`, `pull_request` |
| Signature Verification | `github_service.py` → `verify_webhook_signature()` | HMAC-SHA256 verification using per-project or global secrets |
| Auto Re-ingestion | `main.py` → `_sync_project_background()` | On push/issue/PR events, a **background task** automatically re-ingests the full repo context |
| Webhook Registration | `github_service.py` → `register_webhook()` | Automatically registers webhooks when a project is created |
| Event Audit Log | `models.py` → `GitWebhookEvent` | All webhook events are persisted with type, summary, and processed status |

**Flow**:
```
Developer pushes code to GitHub
         │
         ▼
GitHub sends POST /webhook/github
         │
         ▼
HMAC-SHA256 signature verification (per-project secret)
         │
         ▼
Event logged to git_webhook_events table
         │
         ▼
Background task triggered: _sync_project_background()
   • Deletes old vector store collection
   • Fetches full fresh context (code + issues + PRs + commits + branches)
   • Re-chunks and re-vectorizes with dynamic sizing
   • Creates new ChromaDB collection
   • Updates last_synced_at timestamp
         │
         ▼
Context is now FRESH — next query uses updated data
```

Additionally, users can **manually trigger re-sync** via `POST /projects/{id}/sync` at any time.

---

### R2: Multiple Projects — Support multiple repos/workspaces ✅

**Implementation**: `GitProject` model with per-user, per-repo isolation

| Component | File | How It Works |
|---|---|---|
| Project Model | `models.py` → `GitProject` | Each project stores `repo_url`, `repo_owner`, `repo_name`, `branch`, `vector_db_collection_id` |
| Per-User Isolation | `main.py` → all project endpoints | Every query filters by `user_id` — users only see their own projects |
| Unique Vector Stores | `main.py` → `create_vector_store()` | Each project gets a unique ChromaDB collection: `{user_id}_{timestamp}` |
| Duplicate Prevention | `main.py` → `create_project()` | Checks for existing `(user_id, repo_owner, repo_name)` before creating |

**Endpoints**:
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/projects` | List all Git projects for the authenticated user |
| `POST` | `/projects` | Add a new GitHub repo — ingests full context + registers webhook |
| `DELETE` | `/projects/{id}` | Delete project and its vector store collection |
| `POST` | `/projects/{id}/sync` | Manually re-sync a project (background task) |

Each user can manage **unlimited repositories**, each with its own independent vector store, webhook, and sync state.

---

### R3: Persistent Sessions — Return and continue where you left off ✅

**Implementation**: PostgreSQL-backed chat and message persistence

| Component | File | How It Works |
|---|---|---|
| Chat Model | `models.py` → `Chat` | Stores `title`, `type` (git_chat), `vector_db_collection_id` linking to the project |
| Message Model | `models.py` → `Message` | Every user and assistant message is persisted with `role` and `content` |
| Chat History API | `main.py` → `GET /chats/{id}/messages` | Retrieves full conversation history ordered by `created_at` |
| User Isolation | `main.py` → all chat endpoints | All chats filtered by `user_id` — complete data isolation |

**Flow**:
```
User sends message → saved to messages table (role="user")
         │
         ▼
AI responds via streaming → full response saved (role="assistant")
         │
         ▼
User closes browser / logs out
         │
         ▼
User returns later → GET /chats lists all previous sessions
         │
         ▼
User opens a chat → GET /chats/{id}/messages restores full history
```

Every message (both user and assistant) is persisted to PostgreSQL immediately after generation, ensuring **zero data loss**.

---

### R4: Usable UI — Clear, functional interface ✅

**Implementation**: Next.js 15 frontend with dedicated Git features

| Feature | Component | Description |
|---|---|---|
| Git Projects Panel | `git-projects-panel.tsx` | Full CRUD for managing GitHub repos — add, delete, sync, generate reports |
| Chat Interface | `chat-view.tsx` + `chat-messages.tsx` | Real-time streaming chat with markdown rendering |
| Source Selection | `source-input-dialog.tsx` | Dialog for entering GitHub repo URLs |
| Sidebar Navigation | `app-sidebar.tsx` | Chat history list, type icons, and navigation |
| Dark/Light Theme | `theme-switcher.tsx` | Toggle between themes |
| Responsive Layout | `mobile-menu.tsx` | Mobile-friendly with collapsible sidebar |

---

## 🔗 Core Pipeline Implementation — Backend Deep Dive

### Step 1: Repo Context Ingestion (`github_service.py`)

The `build_full_context()` function orchestrates **5 parallel fetchers** to build a comprehensive text representation of the entire repository:

| Fetcher | Data | Limits | Details |
|---|---|---|---|
| `fetch_repo_code()` | Source files | All matching files | LangChain `GithubFileLoader` — supports **40+ file extensions** |
| `fetch_repo_issues()` | Issues | Up to 100 | Labels, bodies (truncated 2000 chars), up to 5 comments each |
| `fetch_repo_pull_requests()` | Pull Requests | Up to 50 | Status, merge info, changed files with +/- stats |
| `fetch_repo_commits()` | Commit History | Up to 100 | Author, message (first line), SHA, stats |
| `fetch_repo_branches()` | Branches | All | Default/protected markers |

**Supported File Extensions**:
| Category | Extensions |
|---|---|
| **Text/Docs** | `.txt`, `.md`, `.html`, `.css`, `.xml`, `.json`, `.yaml`, `.yml` |
| **Python/JS/TS** | `.py`, `.js`, `.ts`, `.jsx`, `.tsx` |
| **JVM** | `.java`, `.kt`, `.kts`, `.scala` |
| **Systems** | `.c`, `.cpp`, `.h`, `.hpp`, `.rs`, `.go`, `.swift` |
| **Others** | `.php`, `.rb`, `.lua`, `.sh`, `.bash`, `.r`, `.jl`, `.dart`, `.cs` |
| **Config** | `.toml`, `.cfg`, `.ini`, `.env.example`, `Dockerfile`, `Makefile` |
| **Notebooks** | `.ipynb` |

### Step 2: Vectorization (ChromaDB + Mistral Embeddings)

```
Full repo context text (code + issues + PRs + commits + branches)
      │
      ▼
   Dynamic Chunk Sizing (get_dynamic_chunk_size)
   Adapts chunk_size and overlap based on total document length:
      │
      │   < 1K chars    → chunk = length/2,   overlap = 20
      │   < 5K chars    → chunk = length/5,   overlap = 50
      │   < 20K chars   → chunk = length/20,  overlap = 100
      │   < 100K chars  → chunk = length/80,  overlap = 200
      │   < 300K chars  → chunk = length/200, overlap = 400
      │   ≥ 300K chars  → chunk = 6000,       overlap = 600
      │
      ▼
   RecursiveCharacterTextSplitter → text chunks
      │
      ▼
   MistralAIEmbeddings (model: "mistral-embed") → vector embeddings
      │
      ▼
   ChromaDB persisted collection (name: {user_id}_{timestamp_ms})
```

### Step 3: Searchable (MMR Retriever)

When a user asks a question, the system uses **Maximal Marginal Relevance (MMR)** retrieval:
- Retrieves `k=5` most relevant **and diverse** document chunks from the vector store
- Avoids redundancy in retrieved context
- Assembles context text for the LLM prompt

### Step 4: Report Generation (`/git_report`)

AI-powered structured reports with **4 specialized prompt templates**:

| Report Type | Focus Areas |
|---|---|
| `full` | Overview, architecture, dependencies, code quality, activity, recommendations |
| `architecture` | System architecture, module structure, design patterns, data flow, API design |
| `dependencies` | Core/dev dependencies, health, security concerns, optimization |
| `code_quality` | Organization, best practices, error handling, testing, technical debt |

**Report Generation Flow**:
1. Load associated vector store
2. Execute **5 diverse semantic queries** with MMR retriever (`k=20`)
3. De-duplicate retrieved chunks across all queries
4. Apply report-type-specific prompt template
5. Generate report via `ChatMistralAI` (mistral-small-latest)

### Step 5: Conversational (`/chat/stream`)

RAG-grounded streaming chat for Git repositories:

```
User query: "How is authentication implemented in this repo?"
      │
      ▼
   Load project's ChromaDB vector store
      │
      ▼
   MMR Retriever → 5 most relevant chunks from repo context
      │
      ▼
   RAG Prompt (anti-hallucination, grounded-only answers) + context + question
      │
      ▼
   ChatMistralAI (mistral-small-latest, temp=0.3, streaming=true)
      │
      ▼
   StreamingResponse → token-by-token to frontend
      │
      ▼
   Full response saved to PostgreSQL (role="assistant")
```

The RAG prompt enforces **strict grounding rules** — the model must only answer from repo context and explicitly state when information is insufficient, preventing hallucination.

---

## 🌟 Extra Features (Beyond Problem Requirements)

Our backend goes **significantly beyond** the base requirements with the following additional features:

### 1. 🧠 Multi-Source RAG (Not Just GitHub)

The platform supports **4 data sources**, not just GitHub:

| Source | Endpoint | Loader | Description |
|---|---|---|---|
| **YouTube** | `POST /yt_rag` | `youtube-transcript-api` | Extracts video transcript → vectorizes |
| **PDF** | `POST /pdf_rag` | `PyPDFLoader` | Parses PDF pages → vectorizes |
| **Webpage** | `POST /web_rag` | `WebBaseLoader` + BeautifulSoup | Scrapes page text → vectorizes |
| **GitHub** | `POST /git_rag` | `GithubFileLoader` + PyGithub | Full repo context → vectorizes |

Each source type creates its own ChromaDB vector store and powers a dedicated chat type.

### 2. 💬 Normal Chat Mode (Non-RAG)

Beyond RAG-based conversations, the platform includes a **free-form AI chat** mode using `ConversationBufferMemory` from LangChain — providing a full conversational AI experience without any external data source.

### 3. 🏷️ AI-Powered Auto Title Generation (Parallel Execution)

On the **first message** of any chat, the system uses **parallel execution**:
- A **background thread** generates a concise title (max 5 words) using Mistral AI
- The **main thread** streams the response simultaneously
- After streaming, a `<!-- TITLE_UPDATE:title -->` marker is sent to the frontend for real-time sidebar updates

No waiting — title and response are generated in parallel.

### 4. 🔐 Full JWT Authentication System

| Feature | Implementation |
|---|---|
| Password Hashing | SHA-256 pre-hash → Base64 → bcrypt (avoids bcrypt's 72-byte limit) |
| JWT Tokens | HS256 algorithm, 30-day expiration |
| Token Extraction | `HTTPBearer` with manual header fallback |
| Per-User Data Isolation | All queries filtered by `user_id` from decoded JWT |

### 5. 📊 5 Chat Types Instead of Just Git

| Chat Type | Key | Description |
|---|---|---|
| Normal | `normal_chat` | Free-form AI conversation |
| YouTube | `yt_chat` | Chat grounded in YouTube transcript |
| PDF | `pdf_chat` | Chat grounded in PDF content |
| Web | `web_chat` | Chat grounded in webpage content |
| Git | `git_chat` | Chat grounded in GitHub repo context |

### 6. 🔒 HMAC-SHA256 Webhook Signature Verification

Webhooks don't just trigger blindly — every incoming webhook is verified using **HMAC-SHA256 signature verification** with per-project secrets, preventing unauthorized trigger of re-ingestion.

### 7. 📈 Dynamic Chunk Sizing

Unlike static chunking, the system **dynamically calculates** optimal `chunk_size` and `chunk_overlap` based on document length — ensuring small documents aren't over-fragmented and large repositories are chunked efficiently.

### 8. 🛡️ Anti-Hallucination RAG Prompt Engineering

The RAG prompt template enforces strict grounding rules:
- Only answer from provided context
- Explicitly say "I don't have enough information" when context is insufficient
- Never invent facts not in the context
- Adaptive response length based on user needs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 15)                    │
│  Landing Page ─── Auth (Sign In/Up) ─── Chat Interface          │
│  Git Projects Panel ─── Report Viewer ─── Markdown Renderer     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API + Streaming (HTTP)
┌───────────────────────────▼─────────────────────────────────────┐
│                     BACKEND (FastAPI 2.0.0)                     │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Auth   │  │ Chat Manager │  │ RAG Ingestion│              │
│  │ (JWT/    │  │ (5 types,    │  │ (YT, PDF,    │              │
│  │  bcrypt) │  │  streaming)  │  │  Web, Git)   │              │
│  └──────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐         │
│  │ Git Project  │  │   GitHub     │  │    Report     │         │
│  │  Management  │  │   Webhooks   │  │   Generator   │         │
│  └──────────────┘  └──────────────┘  └───────────────┘         │
└───┬─────────────────┬──────────────────┬────────────────────────┘
    │                 │                  │
    ▼                 ▼                  ▼
┌────────┐    ┌────────────┐    ┌──────────────┐
│PostgreSQL│   │  ChromaDB  │    │  Mistral AI  │
│ (Users, │   │ (Vectors,  │    │  (LLM +      │
│  Chats, │   │  Embeddings│    │   Embeddings)│
│  Msgs)  │   │  per user) │    │              │
└────────┘    └────────────┘    └──────────────┘
```

---

## 🛠️ Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **FastAPI** | Async web framework with auto OpenAPI docs |
| **LangChain** | Orchestration — chains, prompts, memory, retrievers |
| **Mistral AI** | LLM (`mistral-small-latest`) + Embeddings (`mistral-embed`) |
| **ChromaDB** | Local vector database for RAG embeddings |
| **PostgreSQL** | Relational DB for users, chats, messages, projects |
| **SQLAlchemy** | ORM and database session management |
| **PyGithub** | GitHub API interaction (code, issues, PRs, webhooks) |
| **python-jose** | JWT token encoding/decoding |
| **bcrypt** | Password hashing with SHA-256 pre-hash |
| **BeautifulSoup4** | Web page content extraction |
| **PyPDF** | PDF document parsing |
| **youtube-transcript-api** | YouTube transcript extraction |
| **Uvicorn / Gunicorn** | ASGI server |

### Frontend

| Technology | Purpose |
|---|---|
| **Next.js 15** | React framework with Turbopack dev server |
| **TypeScript** | Type-safe frontend development |
| **TailwindCSS** | Utility-first CSS styling |
| **Radix UI** | Accessible headless component primitives (35 components) |
| **Firebase** | Cloud services integration |
| **GenKit AI** | Google AI flow orchestration |
| **Recharts** | Data visualization charts |
| **Marked + KaTeX** | Markdown & LaTeX rendering |

---

## 📂 Backend Project Structure

```
Sonyc_Backend/
├── app/
│   ├── __init__.py              # Package initializer
│   ├── main.py                  # Core application — all endpoints, RAG logic, streaming (1690 lines)
│   ├── auth.py                  # JWT authentication, password hashing (SHA-256 + bcrypt)
│   ├── models.py                # SQLAlchemy ORM models (User, Chat, Message, GitProject, GitWebhookEvent)
│   ├── database.py              # PostgreSQL engine, session factory, dependency injection
│   └── github_service.py        # GitHub API service — context fetchers + webhook helpers
├── requirements.txt             # Python dependencies
├── .gitignore
├── patch_chromadb.py            # ChromaDB compatibility patches
├── debug_github_token.py        # GitHub token debugging utility
└── <uuid-directories>/          # ChromaDB persistent vector store collections
```

---

## 📡 Complete API Reference

| Category | Method | Endpoint | Auth | Description |
|---|---|---|---|---|
| **Home** | `GET` | `/` | ❌ | API welcome message |
| **Debug** | `GET` | `/debug_token` | ❌ | Check GitHub token status |
| **Auth** | `POST` | `/auth/signup` | ❌ | Register new user, returns JWT |
| **Auth** | `POST` | `/auth/signin` | ❌ | Login, returns JWT |
| **Auth** | `GET` | `/auth/me` | ✅ | Get current user info |
| **Chats** | `GET` | `/chats` | ✅ | List all user chats |
| **Chats** | `POST` | `/chats` | ✅ | Create a new chat session |
| **Chats** | `GET` | `/chats/{id}/messages` | ✅ | Get chat message history |
| **Chats** | `DELETE` | `/chats/{id}` | ✅ | Delete a chat |
| **Streaming** | `POST` | `/chat/stream` | ✅ | Send message, receive streamed response |
| **RAG** | `POST` | `/yt_rag` | ✅ | Ingest YouTube video transcript |
| **RAG** | `POST` | `/pdf_rag` | ✅ | Ingest PDF document |
| **RAG** | `POST` | `/web_rag` | ✅ | Ingest webpage content |
| **RAG** | `POST` | `/git_rag` | ✅ | Ingest GitHub repo (full context) |
| **Projects** | `GET` | `/projects` | ✅ | List all Git projects |
| **Projects** | `POST` | `/projects` | ✅ | Add repo + ingest + register webhook |
| **Projects** | `DELETE` | `/projects/{id}` | ✅ | Delete project + vector store |
| **Projects** | `POST` | `/projects/{id}/sync` | ✅ | Re-sync project (background) |
| **Webhook** | `POST` | `/webhook/github` | HMAC | Receive GitHub webhook events |
| **Reports** | `POST` | `/git_report` | ✅ | Generate AI report from project |

> **Docs**: FastAPI auto-generates interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 20+**
- **PostgreSQL** (local or remote instance)
- **Mistral AI API Key** — [console.mistral.ai](https://console.mistral.ai/)
- **GitHub Personal Access Token** — [github.com/settings/tokens](https://github.com/settings/tokens)

### 1. Clone the Repository

```bash
git clone https://github.com/M-Azfar9/sonyc_git_rag.git
cd sonyc_git_rag
```

### 2. Backend Setup

```bash
cd Sonyc_Backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file (see Environment Variables section)
# Add: DATABASE_URL, MISTRAL_API_KEY, JWT_SECRET_KEY, GITHUB_PERSONAL_ACCESS_TOKEN

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000` | Swagger docs at `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd Sonyc_Frontend

# Install dependencies
npm install

# Create .env.local
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local

# Run the development server
npm run dev
```

Frontend available at `http://localhost:3000`

---

## ⚙️ Environment Variables

### Backend (`Sonyc_Backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | `postgresql://postgres:postgres@localhost:5432/sonyc_db` | PostgreSQL connection string |
| `MISTRAL_API_KEY` | ✅ | — | Mistral AI API key for LLM and embeddings |
| `JWT_SECRET_KEY` | ✅ | `your-secret-key-change-in-production` | Secret for JWT signing |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | For Git features | — | GitHub PAT for repo access |
| `CORS_ORIGINS` | ❌ | `http://localhost:3000,...` | Comma-separated allowed origins |
| `WEBHOOK_CALLBACK_URL` | ❌ | — | Public URL for GitHub webhook callbacks |
| `GITHUB_WEBHOOK_SECRET` | ❌ | — | Global webhook signature verification |

### Frontend (`Sonyc_Frontend/.env.local`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | ✅ | `http://localhost:8000` | Backend API base URL |

---

## 🐳 Deployment

### Frontend (Docker)

```bash
cd Sonyc_Frontend
docker build -t sonyc-frontend .
docker run -p 3000:3000 sonyc-frontend
```

Multi-stage Dockerfile: `node:20-alpine` build → standalone Next.js → non-root user.

### Backend

```bash
cd Sonyc_Backend

# Production with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📄 License

This project is developed as part of the **Module 2 DevCon** coursework.

---

<p align="center">
  Built with ❤️ using FastAPI, LangChain, Mistral AI, ChromaDB, and Next.js
</p>
