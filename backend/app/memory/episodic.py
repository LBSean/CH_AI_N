"""
Episodic memory — store and retrieve per-user conversation summaries.

Store path (Phase 2):
    Celery task summarize_episode()
      → LLM generates summary
      → embed summary
      → INSERT INTO episodic_memory

Query path (Phase 3, memory_injection node):
    query_episodic_memory(user_id, query, top_k)
      → embed query
      → cosine similarity search over episodic_memory
      → return formatted context string
"""

from app.core.db import get_db


async def query_episodic_memory(user_id: str, query: str, top_k: int = 3) -> str:
    """
    Find the most relevant past-session summaries for the current query.
    Returns a formatted string for injection into the system prompt,
    or empty string if no relevant episodes exist.
    """
    from app.core.config import get_settings
    settings = get_settings()

    # Embed the query
    embedding = await _embed(query, settings)
    if embedding is None:
        return ""

    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT summary,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   episodic_memory
                WHERE  user_id = %s
                  AND  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (_vec_literal(embedding), user_id, _vec_literal(embedding), top_k),
            )
            rows = await cur.fetchall()

    if not rows:
        return ""

    parts = [f"Past session {i+1}: {row['summary']}" for i, row in enumerate(rows)]
    return "\n".join(parts)


async def store_episodic_memory(
    user_id: str,
    conversation_id: str | None,
    summary: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> None:
    """Insert one episodic memory row."""
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO episodic_memory
                    (user_id, conversation_id, summary, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s)
                """,
                (
                    user_id,
                    conversation_id,
                    summary,
                    _vec_literal(embedding),
                    metadata or {},
                ),
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec_literal(v: list[float]) -> str:
    """Convert a float list to a Postgres vector literal string."""
    return "[" + ",".join(str(x) for x in v) + "]"


async def _embed(text: str, settings) -> list[float] | None:
    """Generate an embedding via the configured embedding model."""
    try:
        if settings.litellm_mode == "library":
            import litellm
            response = await litellm.aembedding(
                model=settings.embedding_model,
                input=[text],
            )
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{settings.litellm_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {settings.litellm_master_key}"},
                    json={"model": settings.embedding_model, "input": text},
                    timeout=30,
                )
                resp.raise_for_status()
                response_data = resp.json()
                return response_data["data"][0]["embedding"]

        return response.data[0].embedding
    except Exception:
        return None
