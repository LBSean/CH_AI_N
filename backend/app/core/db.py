"""
Async database helpers for application-level queries (auth, cost tracking, etc.).
Separate from the sync ConnectionPool used by the LangGraph checkpointer.

psycopg3 (psycopg) API — uses cursor.fetchone(), not fetchrow().

Usage:
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = await cur.fetchone()   # returns a dict (dict_row factory)
"""

import psycopg
from psycopg.rows import dict_row
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.core.config import get_settings


@asynccontextmanager
async def get_db() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Yields an async Postgres connection with dict_row as the default row factory.
    Closes cleanly on exit.
    """
    settings = get_settings()
    conn = await psycopg.AsyncConnection.connect(
        settings.database_url,
        autocommit=True,
        row_factory=dict_row,
    )
    try:
        yield conn
    finally:
        await conn.close()


async def set_rls_user(conn: psycopg.AsyncConnection, user_id: str) -> None:
    """
    Set the session-local user for RLS policies.
    Must be called before any queries on user-scoped tables.
    """
    await conn.execute(
        "SELECT set_config('app.current_user_id', %s, true)",
        (str(user_id),),
    )
