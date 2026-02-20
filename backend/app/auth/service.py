"""Database operations for user management."""

from typing import Any

from app.core.db import get_db
from app.auth.security import hash_password, verify_password


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, email, password_hash, plan, token_budget, tokens_used "
                "FROM users WHERE email = %s",
                (email,),
            )
            return await cur.fetchone()


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, email, plan, token_budget, tokens_used "
                "FROM users WHERE id = %s",
                (user_id,),
            )
            return await cur.fetchone()


async def create_user(email: str, password: str) -> dict[str, Any]:
    """
    Insert a new user. Raises psycopg.errors.UniqueViolation if email exists.
    Returns the new user dict with id, email, plan.
    """
    pw_hash = hash_password(password)
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
                "RETURNING id, email, plan",
                (email, pw_hash),
            )
            return await cur.fetchone()


async def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """Return the user dict if credentials are valid, else None."""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


async def increment_tokens_used(user_id: str, tokens: int) -> None:
    """Increment the rolling token counter for budget enforcement."""
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE users SET tokens_used = tokens_used + %s, updated_at = now() "
                "WHERE id = %s",
                (tokens, user_id),
            )
