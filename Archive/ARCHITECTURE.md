# Ch_AI_n — Target Architecture Specification

> Finalized: 2026-02-20
> Status: Approved — ready for phased implementation

---

## Responsibility Map

| Layer | Technology | Responsibility |
|-------|-----------|---------------|
| Model Gateway | LiteLLM (library mode) | Model routing, fallback, cost tracking |
| Agent Runtime | LangGraph | Orchestration, state, tools, guardrails |
| Tool Adapters | LangChain | Tool wrappers, embedding adapters only |
| Knowledge | LlamaIndex | Document ingestion, chunking, retrieval |
| Relational Store | PostgreSQL 16 | Users, conversations, messages, episodic memory, tool state, cost tracking |
| Vector Store | pgvector | Semantic memory, document embeddings |
| Cache & Queue | Redis 7 | Celery broker, embedding cache, rate limiting |
| Background Worker | Celery | Async ingestion, summarization, analytics logging |
| Observability | LangSmith + structlog | LLM traces + application structured logging |
| Backend | FastAPI | API, auth, middleware, streaming |
| Frontend | Next.js 15 | UI, SSE proxy, edge auth verification |
| Deployment | Railway + Vercel | Backend on Railway, frontend on Vercel |

---

## 1. Model Layer — LiteLLM (Library Mode)

### Deployment Mode

LiteLLM runs as a **Python library** imported into FastAPI — not as a separate proxy service. This eliminates network overhead on every LLM call while retaining model abstraction, fallback routing, and cost tracking.

**Development:** LiteLLM proxy container remains in docker-compose for local model experimentation and Flowise integration. Backend uses library mode.

**Production:** Library mode only. No proxy container.

### Model Routing

```
Primary LLM:       gpt-4o
Fallback LLM:      claude-sonnet-4-6
Fast/Cheap LLM:    gpt-4o-mini (summaries, validation retries)
Embedding Model:   text-embedding-3-small (1536 dims)
```

Routing strategy: `cost-based-routing` (prefer cheapest model that meets quality threshold).

### Required Capabilities

- **Failover chain:** primary → fallback, automatic on timeout or error
- **Streaming support:** token-by-token SSE delivery
- **Timeout caps:** 60s for chat completions, 30s for embeddings, 10s for classification
- **Cost extraction:** extract `usage` from every response, pass to async cost logger
- **Per-user budget enforcement:** check `users.token_budget` before each LLM call via middleware

### Cost Tracking Flow

```
LLM response → extract usage → FastAPI BackgroundTask → write to cost_tracking table
                                                      → check budget threshold
                                                      → alert if >80% consumed
```

Cost tracking uses FastAPI `BackgroundTasks` (lightweight), not Celery (overkill for a single DB write).

---

## 2. Agent Runtime — LangGraph

### Authority Rule

**LangGraph is the sole orchestration authority.** No orchestration logic lives in LlamaIndex, LangChain, or raw application code. All state transitions, tool routing, retry policies, and guardrails are expressed as LangGraph nodes and edges.

### Graph Architecture

```
START → memory_injection → agent → router → {respond | tool_executor | retrieve | validate}
                                                ↓              ↓            ↓
                                               END      agent (loop)    agent (loop)

validate → {respond | retry_agent | fallback_model}
```

**Nodes:**

| Node | Responsibility |
|------|---------------|
| `memory_injection` | Load RAG context + episodic memory + tool state into system prompt |
| `agent` | LLM reasoning with full context. Produces response or tool call request. |
| `router` | Conditional edge: inspect agent output → route to respond, tool, retrieve, or validate |
| `tool_executor` | Execute tool call, return result to agent for next reasoning step |
| `retrieve` | Additional RAG retrieval when agent requests more context |
| `validate` | Pydantic validation on structured outputs. Retry or fallback on failure. |
| `respond` | Format final response, trigger async logging |

### State Definition

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]    # Full message history
    user_id: str                               # Authenticated user
    thread_id: str                             # Conversation thread
    context: str                               # RAG-injected context
    episodic_context: str                      # Relevant past session summaries
    tool_calls: list                           # Pending tool invocations
    tool_results: dict                         # Completed tool outputs
    metadata: dict                             # Per-run metadata (model, tokens, latency)
    validation_attempts: int                   # Retry counter for structured output
```

### Checkpointing

- PostgresSaver with connection pooling (max 20 connections)
- Thread-based persistence: resume conversations via `thread_id`
- **Cleanup job:** Archive checkpoints older than 30 days after episodic summarization

---

## 3. LangChain — Tool Adapters Only

### Allowed Uses

- `langchain-openai`: ChatOpenAI wrapper for LiteLLM-routed calls
- `langchain-anthropic`: ChatAnthropic wrapper
- Tool wrappers via `@tool` decorator
- `with_structured_output()` for Pydantic-enforced responses
- Embedding adapter interfaces

### Prohibited Uses

- Memory management (LangGraph state handles this)
- Agent orchestration (LangGraph handles this)
- Conversation chains (LangGraph handles this)
- Retrieval chains (LlamaIndex handles this)

---

## 4. Knowledge Layer — LlamaIndex

### Responsibilities

- Document ingestion pipeline (file upload → chunk → embed → store)
- Chunking strategy management
- Embedding generation via LiteLLM
- Hybrid retrieval (vector + keyword)
- Index refresh and re-ingestion

### Must NOT

- Orchestrate conversations
- Control state transitions
- Manage session memory

### Chunking Strategy

| Content Type | Chunk Size | Overlap | Method |
|-------------|-----------|---------|--------|
| Default | 512 tokens | 64 tokens | Fixed window |
| Long-form docs | 1024 tokens | 128 tokens | Semantic (paragraph boundaries) |
| FAQ / structured | 256 tokens | 32 tokens | Fixed window |

Store parent document reference on every chunk for surrounding-context retrieval.

### Retrieval

**Hybrid retrieval:** Combine `VectorIndexRetriever` (pgvector cosine similarity) with `BM25Retriever` (keyword matching) via `QueryFusionRetriever`.

**Multi-tenant filtering:** All queries filter by `user_id`. Documents without a user_id are treated as global/shared knowledge.

**Top-k:** Default 5, configurable per query.

### Index Management

- Track source documents via `document_id` field
- Support re-ingestion (delete old chunks, ingest new)
- Background refresh via Celery task

---

## 5. PostgreSQL Schema

### Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### Tables

#### users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),
    token_budget INT NOT NULL DEFAULT 100000,
    tokens_used INT NOT NULL DEFAULT 0,
    rate_limit INT NOT NULL DEFAULT 60,  -- requests per minute
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

#### conversations

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    thread_id TEXT NOT NULL,              -- links to LangGraph checkpoint
    title TEXT,                           -- auto-generated from first message
    message_count INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ
);
```

#### messages

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    user_id UUID NOT NULL REFERENCES users(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    token_count INT,
    model_used TEXT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Written asynchronously by Celery, not in the request path
```

#### episodic_memory

```sql
CREATE TABLE episodic_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    conversation_id UUID REFERENCES conversations(id),
    summary TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_episodic_embedding ON episodic_memory
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

#### tool_state

```sql
CREATE TABLE tool_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    tool_name TEXT NOT NULL,
    state JSONB NOT NULL DEFAULT '{}',
    expires_at TIMESTAMPTZ,              -- TTL for cache invalidation
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, tool_name)
);
```

#### cost_tracking

```sql
CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    conversation_id UUID REFERENCES conversations(id),
    model_name TEXT NOT NULL,
    tokens_in INT NOT NULL,
    tokens_out INT NOT NULL,
    cost_estimate DECIMAL(10, 6) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

#### llamaindex_documents (updated)

```sql
CREATE TABLE llamaindex_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),   -- NULL = global/shared knowledge
    document_id TEXT,                     -- source document tracking
    text TEXT NOT NULL,
    metadata_ JSONB DEFAULT '{}',
    node_id TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_documents_embedding ON llamaindex_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_documents_user ON llamaindex_documents(user_id);
CREATE INDEX idx_documents_document ON llamaindex_documents(document_id);
```

### Row-Level Security

```sql
-- Enable RLS on all user-scoped tables
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodic_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE cost_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE llamaindex_documents ENABLE ROW LEVEL SECURITY;

-- Policy: users can only access their own data
-- Applied via SET LOCAL app.current_user_id = '<uuid>' in connection setup
CREATE POLICY user_isolation ON conversations USING (user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY user_isolation ON messages USING (user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY user_isolation ON episodic_memory USING (user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY user_isolation ON tool_state USING (user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY user_isolation ON cost_tracking USING (user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY user_isolation ON llamaindex_documents USING (user_id IS NULL OR user_id = current_setting('app.current_user_id')::uuid);
```

### Migration Strategy

Use **Alembic** for schema versioning. Raw `init.sql` for initial bootstrap only.

---

## 6. pgvector Usage

- Store embeddings for: documents, episodic summaries
- **Namespace by user_id** — all queries filter on `user_id`
- Similarity metric: cosine distance
- Index type: HNSW (m=16, ef_construction=64)
- Default top-k: 5 per query
- **At scale (>1M vectors):** Consider partitioning by user_id

---

## 7. Memory Model

### A) Working Memory

| Property | Value |
|----------|-------|
| Location | LangGraph `AgentState` |
| Contents | Last N messages, intermediate reasoning, active task variables, RAG context, episodic context |
| Lifetime | Session only (persisted via checkpointer for resume) |

### B) Episodic Memory

| Property | Value |
|----------|-------|
| Trigger | Conversation ends OR message count exceeds threshold (e.g., 20 messages) |
| Process | 1. Pull message history from checkpoint → 2. Summarize via gpt-4o-mini → 3. Embed summary → 4. Store in `episodic_memory` table |
| Used for | Long-term personalization, cross-session context injection |
| Retrieval | Cosine similarity on summary embeddings, top-3 relevant sessions injected into `memory_injection` node |
| Consolidation | Periodic job re-summarizes old episodic memories into higher-level abstractions |

### C) Semantic Memory

| Property | Value |
|----------|-------|
| Location | pgvector (`llamaindex_documents`) |
| Used for | RAG retrieval, cross-session knowledge recall |
| Access | Via LlamaIndex hybrid retriever, filtered by user_id |

### D) Tool Memory

| Property | Value |
|----------|-------|
| Location | PostgreSQL `tool_state` table (JSONB) |
| Used for | Persisting tool outputs, avoiding re-computation, workflow state |
| TTL | Configurable `expires_at` per tool. Stale entries refreshed on access. |
| Access | Loaded in `memory_injection` node, written by `tool_executor` node |

---

## 8. Validation Layer

### Scope

Only validate **structured outputs** — tool calls, classifications, extracted data. Free-form chat responses are not schema-validated.

### Enforcement Flow

```
agent output → is_structured? → YES → Pydantic validate → PASS → respond
                                                         → FAIL → retry (same model, max 2)
                                                                → FAIL → retry (fallback model, max 1)
                                                                       → FAIL → structured failure object
                    → NO → respond directly
```

### Structured Failure Object

```python
class ValidationFailure(BaseModel):
    status: Literal["validation_failed"]
    raw_output: str
    schema_name: str
    attempts: int
    models_tried: list[str]
    error: str
```

This is routable by LangGraph — the graph can decide to inform the user, try a different approach, or escalate.

### Implementation

Use LangChain's `with_structured_output(PydanticModel)` on the ChatModel for nodes requiring structured responses. This leverages native JSON mode or function calling.

---

## 9. Async Infrastructure

### Celery Configuration

```
Broker:     Redis (same instance as cache)
Backend:    Redis (result storage)
Queues:
  - ingestion:  document processing, embedding generation
  - memory:     episodic summarization, memory consolidation
  - analytics:  (reserved for future batch analytics)
```

### Task Distribution

| Task | Queue | Trigger |
|------|-------|---------|
| Document ingestion + embedding | `ingestion` | POST `/api/ingest/*` |
| Bulk re-embedding | `ingestion` | Admin action or schema change |
| Episodic summarization | `memory` | Conversation end or message threshold |
| Memory consolidation | `memory` | Scheduled (weekly) |
| Checkpoint cleanup | `memory` | Scheduled (daily) |

### What NOT to Use Celery For

- **Cost logging:** Use FastAPI `BackgroundTasks` (single DB write, no retry needed)
- **Message logging:** Use FastAPI `BackgroundTasks` (same reason)
- **Simple cache updates:** Inline in request handler

### Monitoring

Add **Flower** (Celery monitoring dashboard) to docker-compose for task queue visibility.

---

## 10. Observability

### LangSmith

Used for:
- LLM call trace inspection
- Node-level agent debugging
- Prompt iteration and evaluation
- Success/failure callbacks on all LLM calls

### Structured Application Logging (structlog)

Every log entry includes:

```json
{
  "timestamp": "ISO-8601",
  "level": "info|warn|error",
  "event": "agent_invoke|tool_call|auth_fail|rate_limit|...",
  "user_id": "uuid",
  "conversation_id": "uuid",
  "model": "gpt-4o",
  "tokens_in": 150,
  "tokens_out": 340,
  "latency_ms": 1200,
  "tool_name": "web_search",
  "error_type": null
}
```

**Division of responsibility:** LangSmith handles LLM-level traces. structlog handles application-level events (auth, rate limits, errors, task queue status, business logic).

### Cost Dashboard

Queryable from `cost_tracking` table:
- Per-user spend (daily/monthly)
- Per-model cost distribution
- Budget utilization percentage
- Exposed via `/api/admin/metrics`

### Budget Enforcement

```
Pre-request:  Check users.tokens_used < users.token_budget
Post-request: Increment users.tokens_used (async)
At 80%:       Log warning
At 100%:      Reject request with 429 + budget_exceeded error
```

---

## 11. Visual Control Plane

### Flowise

- **Use for:** RAG prototyping, prompt ideation, embedding strategy testing
- **Do NOT** deploy to production
- Connects to LiteLLM proxy (development mode) for model access

### LangGraph Studio

- **Use for:** Runtime state inspection, graph debugging, branch verification
- Configured via `langgraph.json` in backend root
- Available locally and via LangGraph Cloud

---

## 12. App Layer — API Structure

### Backend (FastAPI)

#### Auth Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Create user account |
| POST | `/api/auth/login` | Issue JWT |
| POST | `/api/auth/refresh` | Refresh token |

#### Agent Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/agent/invoke` | Synchronous agent call |
| POST | `/api/agent/stream` | SSE streaming agent call |
| GET | `/api/agent/threads/{thread_id}` | Retrieve conversation state |

#### Ingestion Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ingest/file` | Upload document (queued to Celery) |
| POST | `/api/ingest/text` | Ingest raw text (queued to Celery) |

#### Admin Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/metrics` | Cost + usage dashboard data |
| GET | `/api/admin/users` | User management |

#### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | DB + Redis connectivity check |

### Frontend (Next.js 15)

#### Proxy Routes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | SSE proxy to FastAPI `/api/agent/stream` |
| POST | `/api/upload` | Proxy to FastAPI `/api/ingest/file` |

The Next.js proxy layer provides:
- Edge-side request validation
- CORS handling for production domain
- SSE header management (Cache-Control: no-cache, no-transform)
- Future: edge-side auth token verification before forwarding

Frontend calls FastAPI through Next.js API routes. FastAPI handles all auth verification via JWT middleware.

---

## 13. Security

### Authentication

- **Custom auth** with PostgreSQL `users` table
- Password hashing via `bcrypt` (pgcrypto or Python `passlib`)
- JWT tokens (access + refresh) signed with RS256 or HS256
- FastAPI middleware verifies JWT on all protected routes
- `user_id` extracted from token and set on request context

### Data Isolation

- **Row-Level Security** on all user-scoped tables
- **Vector namespace isolation:** `user_id` filter on all pgvector queries
- Connection-level `SET LOCAL app.current_user_id` for RLS enforcement

### Rate Limiting

- Redis sliding window counter per `user_id`
- Limits based on `users.rate_limit` (plan-dependent)
- Token budget enforcement per `users.token_budget`
- 429 response with `Retry-After` header on limit breach

### Model Safety

- Timeout caps per call type (60s chat, 30s embedding, 10s classification)
- Tool allowlist per user plan
- Input length limits on user messages

### Prompt Injection Mitigation

- Lightweight input filter before messages enter the agent graph
- Separate system prompt from user content in message construction
- Monitor for instruction-override patterns in structured logging

---

## 14. Docker Compose (Development)

```yaml
services:
  postgres:     pgvector/pgvector:pg16    # Port 5432
  redis:        redis:7-alpine            # Port 6379
  litellm:      ghcr.io/berriai/litellm   # Port 4000 (dev proxy only)
  backend:      ./backend (FastAPI)       # Port 8000
  celery:       ./backend (Celery worker) # No exposed port
  flower:       mher/flower               # Port 5555 (Celery monitoring)
  flowise:      flowiseai/flowise         # Port 3001 (optional)
```

### Production Services

| Service | Provider | Notes |
|---------|----------|-------|
| Backend (FastAPI + Celery) | Railway | Single deploy, separate worker process |
| Frontend (Next.js) | Vercel | Edge runtime |
| PostgreSQL | Neon | Managed, pgvector enabled |
| Redis | Upstash | Serverless |
| Observability | LangSmith | SaaS |

---

## 15. Developer Experience

### Testing

- **Framework:** pytest + pytest-asyncio
- **Test database:** Separate Postgres instance or transactional rollback per test
- **Fixtures:** Factory functions for users, conversations, messages
- **Coverage targets:** Auth middleware, agent graph nodes, validation layer

### Schema Versioning

- **Tool:** Alembic
- Raw `init.sql` for initial bootstrap only
- All schema changes go through versioned migrations

### Local Development

- `docker compose up` boots full stack
- Backend hot-reload via uvicorn `--reload` with volume mount
- Seed script to create test users and sample documents
- FastAPI auto-generates OpenAPI docs at `/docs`

---

## 16. Implementation Phases

### Phase 1 — Foundation

- [ ] Expanded PostgreSQL schema (users, conversations, messages, episodic_memory, tool_state, cost_tracking)
- [ ] Alembic migration setup
- [ ] Custom auth system (register, login, JWT, FastAPI middleware)
- [ ] Celery worker setup + docker-compose integration (+ Flower)
- [ ] LiteLLM library mode integration in FastAPI

### Phase 2 — Memory & Validation

- [ ] Episodic memory pipeline (summarize → embed → store via Celery)
- [ ] Tool memory JSONB persistence
- [ ] LLM output validation with Pydantic + retry + fallback
- [ ] Async message + cost logging via BackgroundTasks
- [ ] Budget enforcement middleware

### Phase 3 — Intelligence

- [ ] Agent graph expansion (router, tool executor, validate, memory injection nodes)
- [ ] Hybrid retrieval (BM25 + vector via LlamaIndex QueryFusionRetriever)
- [ ] Multi-tenant document filtering (user_id on llamaindex_documents)
- [ ] Episodic memory injection into agent context

### Phase 4 — Security & Observability

- [ ] Row-Level Security policies on all user-scoped tables
- [ ] Per-user rate limiting (Redis sliding window)
- [ ] Structured logging (structlog)
- [ ] `/api/admin/metrics` endpoint
- [ ] Tool allowlist per user plan
- [ ] Input sanitization / prompt injection filter
- [ ] Checkpoint cleanup job
- [ ] Memory consolidation job

### Phase 5 — Frontend & Polish

- [ ] `/api/upload` proxy route
- [ ] Auth UI (register, login, token management)
- [ ] Admin metrics dashboard
- [ ] Conversation history UI
- [ ] Seed data script for dev environment
- [ ] Test suite (pytest + fixtures)
