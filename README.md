# CH_AI_N

A production-grade AI agent platform built on LangGraph, LlamaIndex, and FastAPI.
Drop-in foundation for any AI-powered application — multi-tenant, memory-aware, fully observable.

---

## Are you ready to build?

**Yes.** The backend is fully operational:

| Capability | Status |
|---|---|
| Auth (register / login / JWT) | ✅ |
| LLM routing with fallback | ✅ |
| Streaming SSE responses | ✅ |
| Multi-turn conversation memory | ✅ |
| RAG with hybrid retrieval (vector + BM25) | ✅ |
| Episodic memory (cross-session recall) | ✅ |
| Tool calling (agentic loop) | ✅ |
| Async document ingestion (Celery) | ✅ |
| Per-user budget + rate limiting | ✅ |
| Cost tracking (per model, per user) | ✅ |
| Structured logging (structlog) | ✅ |
| Admin metrics API | ✅ |
| Postgres Row-Level Security | ✅ |
| Prompt injection filter | ✅ |
| Frontend chat UI | ✅ |
| Frontend auth UI | 🔜 Phase 5 |
| Test suite | 🔜 Phase 5 |

---

## Stack

```
Model Layer       LiteLLM          — model routing, fallback, cost tracking
Agent Runtime     LangGraph        — orchestration, state, tool calling
Tool Adapters     LangChain        — tool wrappers, LLM adapters
Knowledge         LlamaIndex       — ingestion, chunking, hybrid retrieval
Relational Store  PostgreSQL 16    — users, conversations, messages, episodic memory
Vector Store      pgvector         — semantic memory, document embeddings
Cache / Queue     Redis 7          — rate limiting, Celery broker
Background Worker Celery           — async ingestion, episodic summarisation
Observability     LangSmith        — LLM trace inspection
Logging           structlog        — structured JSON application logs
Backend           FastAPI          — API, auth, streaming
Frontend          Next.js 15       — chat UI, SSE proxy
Deployment        Railway + Vercel — backend + frontend
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | Latest | Runs the full dev stack |
| Node.js | 20+ | Frontend dev server |
| Python | 3.11+ | Local tooling (Alembic, etc.) |
| `gh` CLI | Latest | GitHub operations |
| OpenAI API key | — | Required |
| Anthropic API key | — | Optional (fallback model) |
| LangSmith API key | — | Optional (tracing) |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/LBSean/CH_AI_N.git
cd CH_AI_N

cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY, LITELLM_MASTER_KEY, JWT_SECRET at minimum
```

**Minimum required `.env` values:**
```bash
OPENAI_API_KEY=sk-...
LITELLM_MASTER_KEY=sk-any-string-you-choose
JWT_SECRET=$(openssl rand -hex 32)   # or any long random string
```

### 2. Start the stack

```bash
docker compose up --build
```

Services that come up:

| Service | URL | Purpose |
|---|---|---|
| FastAPI backend | http://localhost:8000 | Main API + docs at `/docs` |
| Next.js frontend | http://localhost:3000 | Chat UI |
| LiteLLM proxy | http://localhost:4000 | LLM gateway (dev) |
| Flower | http://localhost:5555 | Celery task monitor |
| Flowise | http://localhost:3001 | RAG prototyping (optional) |

### 3. Start the frontend (separate terminal)

```bash
cd frontend
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

### 4. Register and start building

```bash
# Register a user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "yourpassword"}'

# Response includes access_token — use it on all subsequent requests
```

---

## API Reference

All protected routes require `Authorization: Bearer <access_token>`.

### Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create account → returns JWT pair |
| `POST` | `/api/auth/login` | Login → returns JWT pair |
| `POST` | `/api/auth/refresh` | Refresh access token |

### Agent

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/agent/invoke` | Synchronous agent call |
| `POST` | `/api/agent/stream` | Streaming SSE agent call |
| `GET` | `/api/agent/threads/{thread_id}` | Retrieve conversation state |

**Stream request:**
```bash
curl -X POST http://localhost:8000/api/agent/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LangGraph?", "thread_id": null}'
```

**Resume a conversation** by passing the `thread_id` from the first response:
```json
{ "message": "Tell me more", "thread_id": "abc-123" }
```

### Ingestion

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ingest/file` | Upload a `.txt` file (queued async) |
| `POST` | `/api/ingest/text` | Ingest raw text + metadata |

```bash
# Ingest text
curl -X POST http://localhost:8000/api/ingest/text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "LangGraph is a library for building stateful agents.", "metadata": {}}'

# Returns: {"status": "queued", "task_id": "..."}
# Watch progress at http://localhost:5555
```

### Admin

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/metrics` | Cost, usage, and budget for the current user |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | DB connectivity check |

Full interactive API docs: **http://localhost:8000/docs**

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI provider key |
| `ANTHROPIC_API_KEY` | — | — | Anthropic fallback key |
| `LITELLM_MASTER_KEY` | ✅ | — | Internal API key for LiteLLM gateway |
| `LITELLM_MODE` | — | `proxy` | `proxy` (dev) or `library` (prod, no extra container) |
| `JWT_SECRET` | ✅ | — | Secret for signing JWTs — use `openssl rand -hex 32` |
| `DATABASE_URL` | ✅ | set in compose | PostgreSQL connection string |
| `REDIS_HOST` | — | `redis` | Redis hostname |
| `REDIS_PASSWORD` | — | — | Redis password (Upstash in prod) |
| `LANGCHAIN_API_KEY` | — | — | LangSmith key — tracing disabled if absent |
| `LANGCHAIN_PROJECT` | — | `ai-workspace` | LangSmith project name |

---

## Project Structure

```
CH_AI_N/
├── backend/                    # FastAPI + LangGraph + LlamaIndex
│   ├── app/
│   │   ├── agents/
│   │   │   ├── nodes.py        # memory_injection, agent, router
│   │   │   ├── research_agent.py  # graph topology
│   │   │   └── tools.py        # LangChain tool definitions
│   │   ├── api/
│   │   │   ├── admin.py        # /api/admin/metrics
│   │   │   ├── agent.py        # /api/agent/* endpoints
│   │   │   ├── health.py       # /health
│   │   │   └── ingest.py       # /api/ingest/*
│   │   ├── auth/
│   │   │   ├── deps.py         # get_current_user dependency
│   │   │   ├── models.py       # Pydantic schemas
│   │   │   ├── router.py       # /api/auth/* endpoints
│   │   │   ├── security.py     # bcrypt + JWT
│   │   │   └── service.py      # DB operations
│   │   ├── core/
│   │   │   ├── checkpointer.py # LangGraph Postgres checkpointer
│   │   │   ├── config.py       # Settings (pydantic-settings)
│   │   │   ├── db.py           # Async psycopg3 helper
│   │   │   ├── graph_state.py  # AgentState TypedDict
│   │   │   ├── llm.py          # LLM factory (proxy/library mode)
│   │   │   └── logging.py      # structlog configuration
│   │   ├── memory/
│   │   │   ├── episodic.py     # query + store episodic memory
│   │   │   └── tool_state.py   # JSONB tool state persistence
│   │   ├── middleware/
│   │   │   ├── budget.py       # Token budget enforcement
│   │   │   ├── rate_limit.py   # Redis sliding-window rate limiter
│   │   │   └── sanitize.py     # Prompt injection filter
│   │   ├── rag/
│   │   │   └── pipeline.py     # Hybrid retrieval (vector + BM25)
│   │   ├── workers/
│   │   │   ├── celery_app.py   # Celery instance + queue config
│   │   │   └── tasks.py        # ingest, summarize_episode, cleanup
│   │   └── main.py             # FastAPI app entrypoint
│   ├── alembic/                # Schema migrations
│   ├── alembic.ini
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                   # Next.js 15 + React 19
│   └── src/
│       ├── app/
│       │   ├── api/chat/       # SSE proxy → FastAPI
│       │   └── page.tsx
│       └── components/
│           └── Chat.tsx
├── docker/
│   └── postgres/
│       └── init.sql            # Full schema + RLS policies
├── litellm/
│   └── config.yaml             # Model routing config (dev proxy)
├── Archive/                    # Architecture + deployment specs
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Development Guide

### Adding a new tool

1. Define it in [backend/app/agents/tools.py](backend/app/agents/tools.py):
```python
@tool
async def my_tool(param: str) -> str:
    """Describe what this tool does — the LLM reads this docstring."""
    return result
```
2. Add it to `ALL_TOOLS`. It's automatically available to the agent.

### Adding a new API route

1. Create `backend/app/api/my_feature.py`
2. Define a router with the standard pattern:
```python
router = APIRouter(prefix="/api/my-feature", tags=["my-feature"])

@router.get("/")
async def my_endpoint(user: CurrentUser = Depends(check_rate_limit)):
    ...
```
3. Register in [backend/app/main.py](backend/app/main.py): `app.include_router(my_feature.router)`

### Triggering episodic summarization

After a conversation ends, call:
```python
from app.workers.tasks import summarize_episode
summarize_episode.delay(conversation_id="...", user_id="...")
```
The Celery worker will summarize the conversation and store it — the agent will recall it automatically on the user's next session.

### Running schema migrations

```bash
# Inside the backend container or with DATABASE_URL set:
cd backend
alembic upgrade head

# Generate a new migration after schema changes:
alembic revision --autogenerate -m "describe your change"
```

### Monitoring tasks

Open **http://localhost:5555** (Flower) to see Celery task queues, retries, and status.

---

## Deployment

| Service | Provider | Notes |
|---|---|---|
| Backend + Celery | Railway | Single deploy, separate worker command |
| Frontend | Vercel | Auto-deploys on push to `main` |
| PostgreSQL | Neon | Managed, pgvector enabled |
| Redis | Upstash | Serverless |
| LiteLLM | N/A in prod | Use `LITELLM_MODE=library` — no container needed |

**Production environment changes:**
```bash
LITELLM_MODE=library          # eliminates the proxy container + network hop
DATABASE_URL=<neon-url>       # managed Postgres
REDIS_HOST=<upstash-host>
REDIS_PASSWORD=<upstash-pass>
JWT_SECRET=<openssl rand -hex 32>
ENVIRONMENT=production
```

CI/CD via GitHub Actions is configured in `.github/workflows/deploy.yml`.

---

## Roadmap

- **Phase 5:** Frontend auth UI, `/api/upload` route, admin dashboard, dev seed data, pytest suite
- **Future:** Scheduled Celery Beat (weekly memory consolidation, daily checkpoint cleanup), multi-agent support, tool allowlist per plan, pgvector partitioning at scale
