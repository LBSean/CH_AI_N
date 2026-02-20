# DEPLOYMENT.md
## Full-Stack Multi-Agent AI Platform

> **Stack:** LiteLLM · LangGraph · LangChain · LlamaIndex · Postgres/pgvector · Redis · LangSmith · Flowise · LangGraph Studio · Next.js/Vercel

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser                                                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ HTTPS
┌─────────────────────────▼───────────────────────────────────────────┐
│  Next.js App (Vercel)                                               │
│  • React UI  • API Routes (proxy)  • Clerk Auth  • SSE consumer    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ HTTP / SSE
┌─────────────────────────▼───────────────────────────────────────────┐
│  FastAPI Backend  (Railway)                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  LangGraph  (agent orchestration + state machine)             │ │
│  │    └── LangChain tools  (search, APIs, code, custom)          │ │
│  │    └── LlamaIndex  (RAG pipeline + retrieval)                 │ │
│  │    └── Postgres Checkpointer  (thread state persistence)      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────┬─────────────────────────┬───────────────────────────────────┘
         │ OpenAI-compatible API   │ pgvector queries
┌────────▼──────────────┐  ┌───────▼───────────────────────────────┐
│  LiteLLM Proxy        │  │  Postgres + pgvector  (Neon)          │
│  (Railway)            │  │  • Documents / embeddings (LlamaIndex)│
│  • Model routing      │  │  • Agent thread state (LangGraph)     │
│  • API key mgmt       │  │  • Relational app data                │
│  • Semantic cache     │  └───────────────────────────────────────┘
│  • Cost tracking      │
│  • Fallbacks          │  ┌───────────────────────────────────────┐
└────────┬──────────────┘  │  Redis  (Upstash)                     │
         │                 │  • LiteLLM semantic cache             │
┌────────▼────────────┐    │  • Session state / rate limits        │
│  LLM Providers      │    └───────────────────────────────────────┘
│  • OpenAI           │
│  • Anthropic        │  ┌───────────────────────────────────────┐
│  • GitHub Copilot   │  │  LangSmith  (SaaS)                    │
│  • (any provider)   │  │  • Traces · Evals · Datasets          │
└─────────────────────┘  └───────────────────────────────────────┘

Side tools:
  Flowise       (Railway)  — prototype / low-code MVP builder
  LangGraph Studio  (local / LangGraph Cloud)  — debug / visualize agents
```

---

## Service Deployment Map

| Service | Local Dev | Production |
|---|---|---|
| Postgres + pgvector | Docker (pgvector/pgvector:pg16) | Neon (managed serverless) |
| Redis | Docker (redis:7-alpine) | Upstash (serverless) |
| LiteLLM Proxy | Docker (ghcr.io/berriai/litellm) | Railway |
| FastAPI Backend | `uvicorn` (local) | Railway |
| Flowise | Docker (flowiseai/flowise) | Railway |
| LangGraph Studio | Desktop app | LangGraph Cloud (optional) |
| LangSmith | SaaS | SaaS (smith.langchain.com) |
| Next.js | `npm run dev` | Vercel |

---

## Phase 0 — Prerequisites

### Required Accounts
- [OpenAI](https://platform.openai.com) — or any supported LLM provider
- [Anthropic](https://console.anthropic.com) — optional (for Claude)
- [GitHub](https://github.com) — with Copilot subscription (optional)
- [LangSmith](https://smith.langchain.com) — free tier available
- [Neon](https://neon.tech) — free tier: 512MB Postgres
- [Upstash](https://upstash.com) — free tier: 10k requests/day Redis
- [Railway](https://railway.app) — $5/month hobby plan
- [Vercel](https://vercel.com) — free tier for Next.js

### Required Local Tools
```bash
# Verify versions
python --version          # 3.11+
node --version            # 20+
docker --version          # 24+
docker compose version    # 2.20+
gh --version              # GitHub CLI (for Copilot auth)

# Install if missing
pip install uv            # fast Python package manager
npm install -g pnpm       # fast Node package manager
```

### Project Structure
```
ai-workspace/
├── backend/              # FastAPI + LangGraph + LlamaIndex
│   ├── app/
│   │   ├── agents/       # LangGraph graph definitions
│   │   ├── chains/       # LangChain tool chains
│   │   ├── rag/          # LlamaIndex RAG pipeline
│   │   ├── api/          # FastAPI route handlers
│   │   └── core/         # Config, auth, shared utilities
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/             # Next.js app
│   ├── app/
│   ├── components/
│   └── package.json
├── litellm/
│   └── config.yaml       # LiteLLM model routing config
├── docker-compose.yml    # Full local dev stack
├── .env.example
└── DEPLOYMENT.md
```

---

## Phase 1 — Database Setup (Postgres + pgvector)

### 1.1 Production: Create Neon Project

1. Sign in at [neon.tech](https://neon.tech)
2. Create project: `ai-workspace` | Region: nearest to Railway deployment
3. Copy the connection string: `postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`
4. Neon enables `pgvector` by default on all projects (no extra step needed)

### 1.2 Schema Setup

Run this against your Postgres instance (Neon console SQL editor or `psql`):

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── LlamaIndex document store ──────────────────────────────────────────────
-- LlamaIndex will auto-create this table via PGVectorStore,
-- but creating it explicitly gives you control over the index type.
CREATE TABLE IF NOT EXISTS llamaindex_documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text        TEXT NOT NULL,
    metadata_   JSONB DEFAULT '{}',
    node_id     TEXT,
    embedding   vector(1536)         -- 1536 for text-embedding-3-small
);

-- HNSW index: fast approximate nearest-neighbor search
-- Tune m and ef_construction for your recall/speed tradeoff
CREATE INDEX IF NOT EXISTS idx_llamaindex_embedding
    ON llamaindex_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── LangGraph checkpointer ─────────────────────────────────────────────────
-- Created automatically by PostgresSaver.setup() — shown here for reference
-- The checkpointer creates: checkpoints, checkpoint_writes, checkpoint_blobs
-- You do NOT need to create these manually.

-- ── Application data ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     TEXT NOT NULL,
    thread_id   TEXT,               -- links to LangGraph checkpoint thread
    created_at  TIMESTAMPTZ DEFAULT now(),
    metadata    JSONB DEFAULT '{}'
);
```

### 1.3 Local Dev: Docker

The `postgres` service in `docker-compose.yml` (Phase 7) handles this automatically.
On first start, mount an init script:

```sql
-- docker/postgres/init.sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

---

## Phase 2 — Model Gateway (LiteLLM)

### 2.1 Configuration File

```yaml
# litellm/config.yaml

model_list:
  # ── OpenAI ──────────────────────────────────────────────────────────────────
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # ── Anthropic ────────────────────────────────────────────────────────────────
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-opus
    litellm_params:
      model: anthropic/claude-opus-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  # ── Embeddings (routed through same gateway) ─────────────────────────────────
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

  # ── Fallback chain (LiteLLM tries these in order on failure) ─────────────────
  - model_name: llm-primary
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: llm-fallback
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: simple-shuffle   # or: least-busy, latency-based
  model_group_alias:
    llm: ["llm-primary", "llm-fallback"]  # call "llm" → auto-routes + falls back

litellm_settings:
  # LangSmith tracing for ALL LLM calls passing through the gateway
  success_callback: ["langsmith"]
  failure_callback: ["langsmith"]

  # Semantic cache — requires Redis
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
    password: os.environ/REDIS_PASSWORD
    ttl: 3600           # cache LLM responses for 1 hour
    similarity_threshold: 0.8  # semantic similarity threshold for cache hits

  # Request/response logging
  request_timeout: 60
  drop_params: false

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL  # LiteLLM uses Postgres for spend tracking
  store_model_in_db: true
```

### 2.2 How downstream services call LiteLLM

All services use LiteLLM as if it were the OpenAI API. Only the `base_url` changes:

```python
# LangChain / LangGraph nodes
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url=os.getenv("LITELLM_BASE_URL"),   # http://litellm:4000/v1
    api_key=os.getenv("LITELLM_MASTER_KEY"),
    model="gpt-4o",                           # maps to config.yaml model_name
)

# LlamaIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(
    model="gpt-4o",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_MASTER_KEY"),
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_MASTER_KEY"),
)
```

### 2.3 Deploy to Railway (Production)

```bash
# From your repository root
railway init
railway add --service litellm
railway variables set \
  OPENAI_API_KEY=sk-... \
  ANTHROPIC_API_KEY=sk-ant-... \
  LITELLM_MASTER_KEY=your-master-key \
  REDIS_HOST=your-upstash-host \
  REDIS_PASSWORD=your-upstash-password \
  DATABASE_URL=postgresql://... \
  LANGCHAIN_API_KEY=ls__... \
  LANGCHAIN_TRACING_V2=true \
  LANGCHAIN_PROJECT=ai-workspace

# Railway uses the Dockerfile in litellm/ or you can use the prebuilt image
# Add to railway.toml:
# [deploy]
# startCommand = "litellm --config /app/config.yaml --port 4000"
```

---

## Phase 3 — Backend Service (FastAPI + LangGraph + LlamaIndex)

### 3.1 Python Dependencies

```toml
# backend/pyproject.toml
[project]
name = "ai-workspace-backend"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Core frameworks
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.1.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",

    # LangGraph persistence
    "langgraph-checkpoint-postgres>=2.0.0",

    # LlamaIndex RAG
    "llama-index>=0.11.0",
    "llama-index-vector-stores-postgres>=0.2.0",
    "llama-index-embeddings-openai>=0.2.0",
    "llama-index-llms-openai>=0.2.0",

    # FastAPI server
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sse-starlette>=2.1.0",    # Server-Sent Events for streaming

    # Database
    "psycopg[binary]>=3.2.0",
    "psycopg-pool>=3.2.0",

    # Utilities
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]
```

### 3.2 Core Configuration

```python
# backend/app/core/config.py
import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str

    # LiteLLM (model gateway)
    litellm_base_url: str = "http://litellm:4000/v1"
    litellm_master_key: str

    # LangSmith (automatic when these are set)
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "ai-workspace"

    # App
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 3.3 LangGraph State + Checkpointer

```python
# backend/app/core/graph_state.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Shared state for all agent graphs."""
    messages: Annotated[list, add_messages]  # full message history
    user_id: str
    thread_id: str
    context: str          # RAG context injected by retrieval node
    tool_calls: list      # pending tool invocations
    metadata: dict        # arbitrary per-run metadata
```

```python
# backend/app/core/checkpointer.py
from functools import lru_cache
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from app.core.config import get_settings

@lru_cache
def get_connection_pool() -> ConnectionPool:
    settings = get_settings()
    return ConnectionPool(
        conninfo=settings.database_url,
        max_size=20,
        kwargs={"autocommit": True},
    )

def get_checkpointer() -> PostgresSaver:
    pool = get_connection_pool()
    checkpointer = PostgresSaver(pool)
    # Creates checkpoints, checkpoint_writes, checkpoint_blobs tables
    # Safe to call repeatedly — idempotent
    checkpointer.setup()
    return checkpointer
```

### 3.4 LlamaIndex RAG Pipeline

```python
# backend/app/rag/pipeline.py
from functools import lru_cache
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
import os

def configure_llamaindex():
    """
    Route all LlamaIndex LLM + embedding calls through LiteLLM.
    Call once at application startup.
    """
    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_MASTER_KEY")

    Settings.llm = OpenAI(
        model="gpt-4o",
        base_url=base_url,
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        base_url=base_url,
        api_key=api_key,
        embed_batch_size=100,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

@lru_cache
def get_vector_store() -> PGVectorStore:
    import psycopg2
    from urllib.parse import urlparse

    db_url = urlparse(os.getenv("DATABASE_URL"))
    return PGVectorStore.from_params(
        database=db_url.path.lstrip("/"),
        host=db_url.hostname,
        password=db_url.password,
        port=db_url.port or 5432,
        user=db_url.username,
        table_name="llamaindex_documents",
        embed_dim=1536,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64},
    )

@lru_cache
def get_index() -> VectorStoreIndex:
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

def get_retriever(similarity_top_k: int = 5):
    index = get_index()
    return index.as_retriever(similarity_top_k=similarity_top_k)

async def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant context for a query. Used in LangGraph retrieval nodes."""
    retriever = get_retriever(similarity_top_k=top_k)
    nodes = await retriever.aretrieve(query)
    if not nodes:
        return ""
    return "\n\n---\n\n".join(node.get_content() for node in nodes)
```

### 3.5 LangGraph Agent Definition

```python
# backend/app/agents/research_agent.py
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from app.core.graph_state import AgentState
from app.core.checkpointer import get_checkpointer
from app.rag.pipeline import retrieve_context
import os

def get_llm():
    return ChatOpenAI(
        base_url=os.getenv("LITELLM_BASE_URL"),
        api_key=os.getenv("LITELLM_MASTER_KEY"),
        model="gpt-4o",
        temperature=0.3,
        streaming=True,  # enables token-by-token streaming to the client
    )

async def retrieve_node(state: AgentState) -> dict:
    """Fetch relevant context from pgvector before the agent responds."""
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    context = await retrieve_context(query)
    return {"context": context}

async def agent_node(state: AgentState) -> dict:
    """Main reasoning node — calls LLM with retrieved context."""
    llm = get_llm()
    system_prompt = f"""You are a helpful AI assistant.

Use the following context to inform your response:
{state.get("context", "No additional context available.")}

Thread ID: {state.get("thread_id", "unknown")}
"""
    from langchain_core.messages import SystemMessage
    messages_with_context = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await llm.ainvoke(messages_with_context)
    return {"messages": [response]}

def build_research_graph():
    """Builds the compiled LangGraph with Postgres checkpointing."""
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("agent", agent_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "agent")
    workflow.add_edge("agent", END)

    checkpointer = get_checkpointer()
    return workflow.compile(checkpointer=checkpointer)
```

### 3.6 FastAPI Routes

```python
# backend/app/api/agent.py
import asyncio
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agents.research_agent import build_research_graph
from app.core.config import get_settings, Settings

router = APIRouter(prefix="/api/agent", tags=["agent"])

class InvokeRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str = "anonymous"

@router.post("/invoke")
async def invoke_agent(req: InvokeRequest, settings: Settings = Depends(get_settings)):
    """Synchronous agent invocation — waits for full response."""
    graph = build_research_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=req.message)],
            "user_id": req.user_id,
            "thread_id": thread_id,
            "context": "",
            "tool_calls": [],
            "metadata": {},
        },
        config=config,
    )

    last_message = result["messages"][-1]
    return {
        "thread_id": thread_id,
        "response": last_message.content,
    }

@router.post("/stream")
async def stream_agent(req: InvokeRequest):
    """
    Streaming agent invocation via Server-Sent Events (SSE).
    The Next.js frontend consumes this as a ReadableStream.
    """
    graph = build_research_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        # First event: thread ID so client can resume conversations
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

        async for event in graph.astream_events(
            {
                "messages": [HumanMessage(content=req.message)],
                "user_id": req.user_id,
                "thread_id": thread_id,
                "context": "",
                "tool_calls": [],
                "metadata": {},
            },
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables Nginx buffering (critical for SSE)
        },
    )

@router.get("/threads/{thread_id}")
async def get_thread_state(thread_id: str):
    """Retrieve the full state of a conversation thread."""
    graph = build_research_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if not state:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "state": state.values}
```

```python
# backend/app/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api import agent, ingest
from app.core.config import get_settings
from app.rag.pipeline import configure_llamaindex

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure LlamaIndex once
    configure_llamaindex()
    yield
    # Shutdown: cleanup if needed

app = FastAPI(title="AI Workspace Backend", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 3.7 Document Ingestion Endpoint

```python
# backend/app/api/ingest.py
from fastapi import APIRouter, UploadFile, File
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from app.rag.pipeline import get_vector_store
from llama_index.core.node_parser import SentenceSplitter

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

@router.post("/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded file into the pgvector knowledge base."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
        vector_store=get_vector_store(),
    )
    documents = [Document(text=text, metadata={"filename": file.filename})]
    nodes = await pipeline.arun(documents=documents)
    return {"status": "ok", "nodes_indexed": len(nodes)}

@router.post("/text")
async def ingest_text(payload: dict):
    """Ingest arbitrary text with optional metadata."""
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
        vector_store=get_vector_store(),
    )
    documents = [Document(
        text=payload["text"],
        metadata=payload.get("metadata", {}),
    )]
    nodes = await pipeline.arun(documents=documents)
    return {"status": "ok", "nodes_indexed": len(nodes)}
```

### 3.8 Backend Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (required for psycopg binary)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install uv && uv pip install --system -e .

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## Phase 4 — Observability (LangSmith)

LangSmith requires zero code changes — it traces automatically when env vars are set.

### 4.1 Environment Variables

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=ai-workspace
```

Set these on every service that runs LangChain/LangGraph code:
- Backend (FastAPI)
- LiteLLM (via `success_callback: ["langsmith"]` in config.yaml)

### 4.2 What Gets Traced Automatically

| Trace | Source | What you see |
|---|---|---|
| LangGraph runs | Backend | Full graph execution, state at each node |
| LLM calls | LangGraph nodes | Input, output, token count, latency |
| LLM calls | LiteLLM gateway | All proxied calls, model routing |
| Retrieval | LlamaIndex | Query, retrieved nodes, similarity scores |
| Tool calls | LangChain tools | Tool input/output, errors |

### 4.3 Evaluation Datasets (Optional)

```python
from langsmith import Client

client = Client()

# Create a dataset for regression testing
dataset = client.create_dataset("agent-qa-pairs")
client.create_examples(
    inputs=[{"question": "What is LangGraph?"}],
    outputs=[{"answer": "LangGraph is a library for building stateful agents..."}],
    dataset_id=dataset.id,
)
```

---

## Phase 5 — Visual Control Plane

### 5.1 Flowise (Prototyping)

#### Local Dev
Handled by Docker Compose (Phase 7).

#### Connect Flowise to LiteLLM
When configuring LLM nodes in Flowise:
- **Provider:** OpenAI
- **Base URL:** `http://litellm:4000/v1` (local) or your Railway LiteLLM URL (production)
- **API Key:** your `LITELLM_MASTER_KEY`
- **Model:** any model name from `config.yaml`

This means Flowise prototypes also benefit from LiteLLM routing, caching, and LangSmith tracing — even in no-code mode.

#### Deploy Flowise to Railway (Production)
```bash
railway add --service flowise
railway variables set \
  FLOWISE_USERNAME=admin \
  FLOWISE_PASSWORD=your-secure-password \
  DATABASE_TYPE=postgres \
  DATABASE_HOST=your-neon-host \
  DATABASE_NAME=flowise \
  DATABASE_USER=your-user \
  DATABASE_PASSWORD=your-password \
  LANGCHAIN_TRACING_V2=true \
  LANGCHAIN_API_KEY=ls__...
```

Flowise Dockerfile for Railway:
```dockerfile
FROM flowiseai/flowise
EXPOSE 3001
CMD ["flowise", "start"]
```

### 5.2 LangGraph Studio

#### Local Development (Desktop App)
1. Install: Download from [LangGraph Studio releases](https://github.com/langchain-ai/langgraph-studio)
2. Create `langgraph.json` in your backend project root:

```json
{
  "dockerfile_lines": [],
  "graphs": {
    "research_agent": "./app/agents/research_agent.py:build_research_graph"
  },
  "env": ".env",
  "python_version": "3.11",
  "dependencies": ["."]
}
```

3. Open LangGraph Studio → Open Folder → select `backend/`
4. Studio reads `langgraph.json`, spins up a local server, and provides the visual debugger

#### LangGraph Cloud (Production, Optional)
```bash
pip install langgraph-cli
langgraph build -t my-agent-image    # builds Docker image
langgraph deploy                     # deploys to LangGraph Cloud
```

LangGraph Cloud provides hosted graph execution + the Studio UI in the cloud.
Alternatively, self-host the LangGraph API server on Railway using the same Docker image.

---

## Phase 6 — Frontend (Next.js + Vercel)

### 6.1 Project Setup

```bash
pnpm create next-app frontend --typescript --tailwind --app --src-dir
cd frontend
pnpm add @clerk/nextjs          # auth
pnpm add ai                     # Vercel AI SDK (streaming helpers)
```

### 6.2 Environment Variables

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000        # dev
# NEXT_PUBLIC_API_URL=https://your-backend.railway.app  # prod

NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
```

### 6.3 API Route — SSE Proxy

```typescript
// frontend/src/app/api/chat/route.ts
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const backendUrl = process.env.NEXT_PUBLIC_API_URL;

  // Proxy the SSE stream from FastAPI to the browser
  const upstream = await fetch(`${backendUrl}/api/agent/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  // Forward the SSE stream directly
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}
```

### 6.4 Chat Component — Streaming

```typescript
// frontend/src/components/Chat.tsx
"use client";

import { useState, useRef } from "react";

interface Message { role: "user" | "assistant"; content: string; }

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const threadIdRef = useRef<string | null>(null);

  async function sendMessage() {
    if (!input.trim() || streaming) return;

    const userMessage = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    setStreaming(true);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        thread_id: threadIdRef.current,
      }),
    });

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = JSON.parse(line.slice(6));

        if (data.type === "thread_id") {
          threadIdRef.current = data.thread_id;
        } else if (data.type === "token") {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1].content += data.content;
            return updated;
          });
        }
      }
    }

    setStreaming(false);
  }

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="flex-1 overflow-y-auto space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`p-3 rounded-lg ${
            msg.role === "user" ? "bg-blue-100 ml-8" : "bg-gray-100 mr-8"
          }`}>
            <p className="text-sm font-semibold">{msg.role}</p>
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </div>
        ))}
      </div>
      <div className="flex gap-2 pt-4">
        <input
          className="flex-1 border rounded-lg px-3 py-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask anything..."
          disabled={streaming}
        />
        <button
          onClick={sendMessage}
          disabled={streaming}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
        >
          {streaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
```

### 6.5 Deploy to Vercel

```bash
cd frontend
vercel                          # interactive deploy wizard
# or:
vercel --prod

# Set environment variables in Vercel dashboard:
# NEXT_PUBLIC_API_URL=https://your-backend.railway.app
# NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
# CLERK_SECRET_KEY=sk_...
```

---

## Phase 7 — Local Development (Docker Compose)

### 7.1 docker-compose.yml

```yaml
# docker-compose.yml
version: "3.9"

services:

  # ── Postgres + pgvector ──────────────────────────────────────────────────────
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: aiworkspace
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: devpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d aiworkspace"]
      interval: 5s
      timeout: 5s
      retries: 5

  # ── Redis ────────────────────────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ── LiteLLM Proxy ────────────────────────────────────────────────────────────
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    command: --config /app/config.yaml --port 4000 --detailed_debug
    volumes:
      - ./litellm/config.yaml:/app/config.yaml
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY}
      REDIS_HOST: redis
      REDIS_PASSWORD: ""
      DATABASE_URL: postgresql://admin:devpassword@postgres:5432/aiworkspace
      LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY}
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_PROJECT: ${LANGCHAIN_PROJECT:-ai-workspace}
    ports:
      - "4000:4000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  # ── FastAPI Backend ───────────────────────────────────────────────────────────
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://admin:devpassword@postgres:5432/aiworkspace
      LITELLM_BASE_URL: http://litellm:4000/v1
      LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY}
      LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY}
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_PROJECT: ${LANGCHAIN_PROJECT:-ai-workspace}
      ENVIRONMENT: development
      CORS_ORIGINS: '["http://localhost:3000"]'
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app   # hot-reload in dev
    depends_on:
      postgres:
        condition: service_healthy
      litellm:
        condition: service_started
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  # ── Flowise ──────────────────────────────────────────────────────────────────
  flowise:
    image: flowiseai/flowise
    environment:
      PORT: 3001
      FLOWISE_USERNAME: ${FLOWISE_USERNAME:-admin}
      FLOWISE_PASSWORD: ${FLOWISE_PASSWORD:-flowise123}
      DATABASE_TYPE: postgres
      DATABASE_HOST: postgres
      DATABASE_PORT: 5432
      DATABASE_NAME: aiworkspace
      DATABASE_USER: admin
      DATABASE_PASSWORD: devpassword
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY}
    ports:
      - "3001:3001"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

### 7.2 Init SQL

```sql
-- docker/postgres/init.sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 7.3 Local .env

```bash
# .env  (never commit this)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LITELLM_MASTER_KEY=sk-litellm-dev-key-change-in-production

LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=ai-workspace

FLOWISE_USERNAME=admin
FLOWISE_PASSWORD=your-flowise-password
```

### 7.4 Start Local Stack

```bash
# Start all backend services
docker compose up -d

# Watch logs
docker compose logs -f backend
docker compose logs -f litellm

# Start Next.js frontend (separate terminal)
cd frontend && pnpm dev

# Start LangGraph Studio (separate, after installing desktop app)
# Open LangGraph Studio → select backend/ folder
```

### 7.5 Service URLs (Local)

| Service | URL |
|---|---|
| FastAPI Backend | http://localhost:8000 |
| FastAPI Docs (Swagger) | http://localhost:8000/docs |
| LiteLLM Proxy | http://localhost:4000 |
| LiteLLM Dashboard | http://localhost:4000/ui |
| Flowise | http://localhost:3001 |
| Next.js Frontend | http://localhost:3000 |
| LangSmith | https://smith.langchain.com |

---

## Phase 8 — Production Deployment

### 8.1 Neon (Postgres)

```bash
# 1. Create project at neon.tech
# 2. Copy connection string — format:
#    postgresql://user:pass@ep-xxx-xxx.us-east-1.aws.neon.tech/neondb?sslmode=require

# 3. Run schema setup via Neon SQL console or psql:
psql "postgresql://..." -f docker/postgres/init.sql
psql "postgresql://..." -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Neon branches are useful for staging:
- `main` branch → production
- `dev` branch → staging (branched from main, same schema)

### 8.2 Upstash (Redis)

```bash
# 1. Create database at upstash.com → Redis → Create Database
# 2. Choose the same region as your Railway services
# 3. Copy:
#    REDIS_HOST=xxx.upstash.io
#    REDIS_PORT=6379
#    REDIS_PASSWORD=xxx
```

### 8.3 Railway (Backend Services)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Initialize project
railway init --name ai-workspace

# Deploy LiteLLM
railway add --service litellm
# Point to litellm/ directory with Dockerfile or use image:
# In Railway dashboard: Settings → Source → Docker Image → ghcr.io/berriai/litellm:main-latest
# Start command: litellm --config /app/config.yaml --port 4000

# Set LiteLLM env vars
railway variables --service litellm set \
  OPENAI_API_KEY=sk-... \
  ANTHROPIC_API_KEY=sk-ant-... \
  LITELLM_MASTER_KEY=sk-litellm-prod-... \
  REDIS_HOST=xxx.upstash.io \
  REDIS_PORT=6379 \
  REDIS_PASSWORD=xxx \
  DATABASE_URL="postgresql://...neon.tech/neondb?sslmode=require" \
  LANGCHAIN_API_KEY=ls__... \
  LANGCHAIN_TRACING_V2=true \
  LANGCHAIN_PROJECT=ai-workspace

# Deploy Backend (FastAPI)
railway add --service backend
# In Railway dashboard: Settings → Source → GitHub repo → backend/ directory

railway variables --service backend set \
  DATABASE_URL="postgresql://...neon.tech/neondb?sslmode=require" \
  LITELLM_BASE_URL=https://litellm-production.up.railway.app/v1 \
  LITELLM_MASTER_KEY=sk-litellm-prod-... \
  LANGCHAIN_API_KEY=ls__... \
  LANGCHAIN_TRACING_V2=true \
  LANGCHAIN_PROJECT=ai-workspace \
  ENVIRONMENT=production \
  CORS_ORIGINS='["https://your-app.vercel.app"]'

# Deploy Flowise
railway add --service flowise
railway variables --service flowise set \
  FLOWISE_USERNAME=admin \
  FLOWISE_PASSWORD=your-secure-password \
  DATABASE_TYPE=postgres \
  DATABASE_HOST=xxx.neon.tech \
  DATABASE_PORT=5432 \
  DATABASE_NAME=neondb \
  DATABASE_USER=xxx \
  DATABASE_PASSWORD=xxx \
  DATABASE_SSL=true \
  LANGCHAIN_API_KEY=ls__...
```

### 8.4 Vercel (Frontend)

```bash
cd frontend
vercel

# Set environment variables via Vercel dashboard or CLI:
vercel env add NEXT_PUBLIC_API_URL production
# Value: https://backend-production.up.railway.app

vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production
vercel env add CLERK_SECRET_KEY production

# Deploy
vercel --prod
```

---

## Phase 9 — CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    name: Deploy Backend to Railway
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Railway CLI
        run: npm install -g @railway/cli

      - name: Deploy
        run: railway up --service backend --detach
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}

  deploy-frontend:
    name: Deploy Frontend to Vercel
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup pnpm
        uses: pnpm/action-setup@v3
        with:
          version: 9

      - name: Install dependencies
        run: cd frontend && pnpm install

      - name: Deploy to Vercel
        run: cd frontend && pnpm vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

---

## Phase 10 — Environment Variables Reference

| Variable | Required | Used By | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes* | LiteLLM | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | LiteLLM | Anthropic API key |
| `LITELLM_MASTER_KEY` | Yes | LiteLLM, Backend | LiteLLM gateway auth key |
| `LITELLM_BASE_URL` | Yes | Backend | LiteLLM proxy URL |
| `DATABASE_URL` | Yes | Backend, LiteLLM, Flowise | Postgres connection string |
| `REDIS_HOST` | Yes | LiteLLM | Redis host for caching |
| `REDIS_PORT` | Yes | LiteLLM | Redis port (default 6379) |
| `REDIS_PASSWORD` | Yes | LiteLLM | Redis auth password |
| `LANGCHAIN_API_KEY` | No | Backend, LiteLLM | LangSmith tracing key |
| `LANGCHAIN_TRACING_V2` | No | Backend, LiteLLM | Enable LangSmith (`true`) |
| `LANGCHAIN_PROJECT` | No | Backend, LiteLLM | LangSmith project name |
| `FLOWISE_USERNAME` | Yes | Flowise | Flowise admin username |
| `FLOWISE_PASSWORD` | Yes | Flowise | Flowise admin password |
| `NEXT_PUBLIC_API_URL` | Yes | Next.js | Backend base URL |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Yes | Next.js | Clerk public key |
| `CLERK_SECRET_KEY` | Yes | Next.js | Clerk secret key |
| `CORS_ORIGINS` | Yes | Backend | JSON array of allowed origins |

*At least one LLM provider key must be configured in LiteLLM.

---

## Phase 11 — Health Checks and Monitoring

### 11.1 Service Health Endpoints

```bash
# LiteLLM
curl http://localhost:4000/health
curl http://localhost:4000/health/readiness

# FastAPI
curl http://localhost:8000/docs

# Postgres (via psql)
psql $DATABASE_URL -c "SELECT version();"

# Redis
redis-cli -h $REDIS_HOST -p 6379 -a $REDIS_PASSWORD ping
```

### 11.2 Add Health Route to FastAPI

```python
# backend/app/api/health.py
from fastapi import APIRouter
from app.core.checkpointer import get_connection_pool

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    return {"status": "ok", "database": db_status}
```

### 11.3 LangSmith Dashboard

Key metrics to watch at [smith.langchain.com](https://smith.langchain.com):
- **Trace error rate** — agent failures, LLM errors
- **Latency percentiles** — p50, p95, p99 per graph run
- **Token usage** — cost tracking per agent type
- **Retrieval quality** — relevance scores from LlamaIndex nodes

### 11.4 LiteLLM Dashboard

At `http://litellm:4000/ui` (local) or your Railway URL:
- **Spend tracking** — per model, per key, per day
- **Cache hit rate** — validates Redis semantic caching is working
- **Fallback events** — when primary model fails and fallback is used
- **Request volume** — calls per minute per model

---

## Quick-Start Checklist

```
Local Dev:
  [ ] Install Docker Desktop, Node 20+, Python 3.11+, pnpm
  [ ] Clone repo, run: cp .env.example .env
  [ ] Fill .env with at least OPENAI_API_KEY and LITELLM_MASTER_KEY
  [ ] docker compose up -d
  [ ] cd frontend && pnpm install && pnpm dev
  [ ] Visit http://localhost:3000

Production:
  [ ] Create accounts: Neon, Upstash, Railway, Vercel, LangSmith
  [ ] Neon: Create project, run schema SQL, copy DATABASE_URL
  [ ] Upstash: Create Redis, copy REDIS_HOST + REDIS_PASSWORD
  [ ] Railway: Deploy litellm + backend + flowise, set all env vars
  [ ] Vercel: Deploy frontend, set NEXT_PUBLIC_API_URL to Railway backend URL
  [ ] LangSmith: Copy API key, set LANGCHAIN_API_KEY on Railway services
  [ ] Test: POST to /api/agent/stream, confirm SSE stream reaches browser
  [ ] LangGraph Studio: open backend/ folder, verify graph visualizes correctly
  [ ] Flowise: configure first flow using LiteLLM as the OpenAI provider
```
