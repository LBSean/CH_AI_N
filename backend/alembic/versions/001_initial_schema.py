"""Initial schema — mirrors docker/postgres/init.sql for Alembic baseline.

Run on existing environments that were bootstrapped via init.sql.
Run on fresh environments that skip init.sql (e.g. Neon in production).

Revision ID: 001
Revises:
Create Date: 2026-02-20
"""

revision = "001"
down_revision = None
branch_labels = None
depends_on = None

from alembic import op


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plan          TEXT NOT NULL DEFAULT 'free'
                          CHECK (plan IN ('free', 'pro', 'enterprise')),
            token_budget  INT NOT NULL DEFAULT 100000,
            tokens_used   INT NOT NULL DEFAULT 0,
            rate_limit    INT NOT NULL DEFAULT 60,
            created_at    TIMESTAMPTZ DEFAULT now(),
            updated_at    TIMESTAMPTZ DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            thread_id     TEXT NOT NULL,
            title         TEXT,
            message_count INT DEFAULT 0,
            total_tokens  INT DEFAULT 0,
            started_at    TIMESTAMPTZ DEFAULT now(),
            ended_at      TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user   ON conversations(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conversations_thread ON conversations(thread_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role            TEXT NOT NULL
                            CHECK (role IN ('user', 'assistant', 'system', 'tool')),
            content         TEXT NOT NULL,
            token_count     INT,
            model_used      TEXT,
            latency_ms      INT,
            created_at      TIMESTAMPTZ DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_user         ON messages(user_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
            summary         TEXT NOT NULL,
            embedding       vector(1536),
            metadata        JSONB DEFAULT '{}',
            created_at      TIMESTAMPTZ DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_episodic_embedding
            ON episodic_memory
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS tool_state (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tool_name  TEXT NOT NULL,
            state      JSONB NOT NULL DEFAULT '{}',
            expires_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (user_id, tool_name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_state_user ON tool_state(user_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS cost_tracking (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
            model_name      TEXT NOT NULL,
            tokens_in       INT NOT NULL,
            tokens_out      INT NOT NULL,
            cost_estimate   DECIMAL(10, 6) NOT NULL DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_cost_user    ON cost_tracking(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cost_created ON cost_tracking(created_at)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS llamaindex_documents (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
            document_id TEXT,
            text        TEXT NOT NULL,
            metadata_   JSONB DEFAULT '{}',
            node_id     TEXT,
            embedding   vector(1536),
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON llamaindex_documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_user     ON llamaindex_documents(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_document ON llamaindex_documents(document_id)")

    # Row-Level Security
    for table in ("conversations", "messages", "episodic_memory",
                  "tool_state", "cost_tracking", "llamaindex_documents"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

    for table, column in [
        ("conversations", "user_id"),
        ("messages", "user_id"),
        ("episodic_memory", "user_id"),
        ("tool_state", "user_id"),
        ("cost_tracking", "user_id"),
    ]:
        op.execute(f"""
            CREATE POLICY user_{table} ON {table}
                USING ({column} = current_setting('app.current_user_id', true)::uuid)
        """)

    op.execute("""
        CREATE POLICY user_llamaindex_documents ON llamaindex_documents
            USING (user_id IS NULL
                   OR user_id = current_setting('app.current_user_id', true)::uuid)
    """)


def downgrade() -> None:
    for table in ("llamaindex_documents", "cost_tracking", "tool_state",
                  "episodic_memory", "messages", "conversations", "users"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
