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
    """
    Returns a PostgresSaver checkpointer wired to the shared connection pool.

    PostgresSaver.setup() is idempotent — it creates the three checkpointer
    tables (checkpoints, checkpoint_writes, checkpoint_blobs) if they don't
    exist yet.  Safe to call on every request.
    """
    pool = get_connection_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer
