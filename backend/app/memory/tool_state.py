"""
Tool state — persist and retrieve per-user, per-tool state across sessions.

Used by the tool_executor node to cache expensive tool outputs.
"""

from typing import Any
from datetime import datetime, timezone

from app.core.db import get_db


async def get_tool_state(user_id: str, tool_name: str) -> dict[str, Any] | None:
    """
    Return the stored state for a tool, or None if absent or expired.
    """
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT state FROM tool_state
                WHERE  user_id = %s
                  AND  tool_name = %s
                  AND  (expires_at IS NULL OR expires_at > now())
                """,
                (user_id, tool_name),
            )
            row = await cur.fetchone()
    return row["state"] if row else None


async def set_tool_state(
    user_id: str,
    tool_name: str,
    state: dict[str, Any],
    ttl_seconds: int | None = None,
) -> None:
    """
    Upsert tool state for a user. Optionally set a TTL for cache invalidation.
    """
    expires_at = None
    if ttl_seconds is not None:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO tool_state (user_id, tool_name, state, expires_at, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (user_id, tool_name)
                DO UPDATE SET
                    state      = EXCLUDED.state,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                """,
                (user_id, tool_name, state, expires_at),
            )
