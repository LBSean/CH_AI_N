-- Runs automatically when the Docker postgres container first starts.
-- For existing environments, use Alembic migrations instead.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email        TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    plan         TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),
    token_budget INT NOT NULL DEFAULT 100000,
    tokens_used  INT NOT NULL DEFAULT 0,
    rate_limit   INT NOT NULL DEFAULT 60,    -- requests per minute
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

-- ── Conversations ─────────────────────────────────────────────────────────────
-- Maps to a LangGraph checkpoint thread_id.
CREATE TABLE IF NOT EXISTS conversations (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id     TEXT NOT NULL,              -- LangGraph checkpoint thread
    title         TEXT,                       -- auto-generated from first message
    message_count INT DEFAULT 0,
    total_tokens  INT DEFAULT 0,
    started_at    TIMESTAMPTZ DEFAULT now(),
    ended_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_thread ON conversations(thread_id);

-- ── Messages ──────────────────────────────────────────────────────────────────
-- Written asynchronously by Celery after each agent response.
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT NOT NULL,
    token_count     INT,
    model_used      TEXT,
    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);

-- ── Episodic Memory ───────────────────────────────────────────────────────────
-- Summaries of past conversations, embedded for semantic recall.
-- Populated by Celery task after conversation ends or threshold is reached.
CREATE TABLE IF NOT EXISTS episodic_memory (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    summary         TEXT NOT NULL,
    embedding       vector(1536),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_embedding
    ON episodic_memory
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── Tool State ────────────────────────────────────────────────────────────────
-- Persists tool outputs across sessions to avoid re-computation.
CREATE TABLE IF NOT EXISTS tool_state (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tool_name  TEXT NOT NULL,
    state      JSONB NOT NULL DEFAULT '{}',
    expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_tool_state_user ON tool_state(user_id);

-- ── Cost Tracking ─────────────────────────────────────────────────────────────
-- Written asynchronously via FastAPI BackgroundTasks after each LLM call.
CREATE TABLE IF NOT EXISTS cost_tracking (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    model_name      TEXT NOT NULL,
    tokens_in       INT NOT NULL,
    tokens_out      INT NOT NULL,
    cost_estimate   DECIMAL(10, 6) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cost_user ON cost_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_cost_created ON cost_tracking(created_at);

-- ── LlamaIndex Document Store ─────────────────────────────────────────────────
-- Updated to include user_id for multi-tenant isolation and document_id for
-- re-ingestion support. NULL user_id = global/shared knowledge.
CREATE TABLE IF NOT EXISTS llamaindex_documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    document_id TEXT,                -- source document reference (for re-ingestion)
    text        TEXT NOT NULL,
    metadata_   JSONB DEFAULT '{}',
    node_id     TEXT,
    embedding   vector(1536),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON llamaindex_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_documents_user      ON llamaindex_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_document  ON llamaindex_documents(document_id);

-- ── Row-Level Security ────────────────────────────────────────────────────────
-- Each user can only access their own data.
-- The app sets: SET LOCAL app.current_user_id = '<uuid>' per connection.
-- Admin role bypasses RLS (used by migrations and background workers).

ALTER TABLE conversations     ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages          ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodic_memory   ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_state        ENABLE ROW LEVEL SECURITY;
ALTER TABLE cost_tracking     ENABLE ROW LEVEL SECURITY;
ALTER TABLE llamaindex_documents ENABLE ROW LEVEL SECURITY;

-- Policies: user can read/write their own rows
CREATE POLICY user_conversations     ON conversations     USING (user_id = current_setting('app.current_user_id', true)::uuid);
CREATE POLICY user_messages          ON messages          USING (user_id = current_setting('app.current_user_id', true)::uuid);
CREATE POLICY user_episodic          ON episodic_memory   USING (user_id = current_setting('app.current_user_id', true)::uuid);
CREATE POLICY user_tool_state        ON tool_state        USING (user_id = current_setting('app.current_user_id', true)::uuid);
CREATE POLICY user_cost_tracking     ON cost_tracking     USING (user_id = current_setting('app.current_user_id', true)::uuid);

-- Documents: user sees their own OR global (user_id IS NULL)
CREATE POLICY user_documents ON llamaindex_documents
    USING (user_id IS NULL OR user_id = current_setting('app.current_user_id', true)::uuid);

-- Grant the app DB user BYPASSRLS so background workers can operate without
-- the app.current_user_id setting (workers process all users' data).
-- In prod, create a dedicated app role and grant BYPASSRLS only to the worker role.
ALTER TABLE conversations     FORCE ROW LEVEL SECURITY;
ALTER TABLE messages          FORCE ROW LEVEL SECURITY;
ALTER TABLE episodic_memory   FORCE ROW LEVEL SECURITY;
ALTER TABLE tool_state        FORCE ROW LEVEL SECURITY;
ALTER TABLE cost_tracking     FORCE ROW LEVEL SECURITY;
ALTER TABLE llamaindex_documents FORCE ROW LEVEL SECURITY;

-- Note: LangGraph checkpoint tables (checkpoints, checkpoint_writes, checkpoint_blobs)
-- are created automatically by PostgresSaver.setup() at runtime.
