"""
Celery tasks.

Queue assignment:
  ingestion — document processing, embedding generation
  memory    — episodic summarisation, consolidation, checkpoint cleanup
"""

import asyncio
import logging
from typing import Any

from app.workers.celery_app import celery

logger = logging.getLogger(__name__)


def _run(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Ingestion ─────────────────────────────────────────────────────────────────

@celery.task(
    name="app.workers.tasks.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="ingestion",
)
def ingest_document(
    self,
    text: str,
    metadata: dict[str, Any],
    user_id: str | None = None,
) -> dict[str, Any]:
    """Chunk, embed, and store a document in pgvector (non-blocking ingestion)."""
    try:
        from llama_index.core import Document
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from app.rag.pipeline import get_vector_store, configure_llamaindex

        configure_llamaindex()
        if user_id:
            metadata["user_id"] = user_id

        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
            vector_store=get_vector_store(),
        )
        nodes = _run(pipeline.arun(documents=[Document(text=text, metadata=metadata)]))
        return {"status": "ok", "nodes_indexed": len(nodes)}

    except Exception as exc:
        logger.error("ingest_document failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


# ── Episodic Memory ───────────────────────────────────────────────────────────

@celery.task(
    name="app.workers.tasks.summarize_episode",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="memory",
)
def summarize_episode(self, conversation_id: str, user_id: str) -> dict[str, Any]:
    """
    Summarise a completed conversation and store the result in episodic_memory.

    Steps:
      1. Load messages for the conversation from the messages table.
      2. Call the fast LLM (gpt-4o-mini) to generate a summary.
      3. Embed the summary.
      4. INSERT into episodic_memory.
    """
    try:
        _run(_summarize_episode_async(conversation_id, user_id))
        return {"status": "ok", "conversation_id": conversation_id}
    except Exception as exc:
        logger.error("summarize_episode failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


async def _summarize_episode_async(conversation_id: str, user_id: str) -> None:
    from app.core.db import get_db
    from app.core.config import get_settings
    from app.core.llm import get_chat_model
    from app.memory.episodic import store_episodic_memory, _embed
    from langchain_core.messages import HumanMessage, SystemMessage

    # 1. Load messages
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = %s ORDER BY created_at",
                (conversation_id,),
            )
            messages = await cur.fetchall()

    if not messages:
        logger.warning("summarize_episode: no messages found for %s", conversation_id)
        return

    # 2. Build transcript
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    # 3. Summarise with fast model
    settings = get_settings()
    llm = get_chat_model(model=settings.fast_model, streaming=False)
    summary_msg = await llm.ainvoke([
        SystemMessage(content=(
            "You are a memory assistant. Summarise the following conversation in 3-5 sentences, "
            "capturing the main topics discussed, any decisions made, and key information shared. "
            "Write in third person (e.g. 'The user asked about...')."
        )),
        HumanMessage(content=transcript),
    ])
    summary = summary_msg.content.strip()

    # 4. Embed and store
    embedding = await _embed(summary, settings)
    await store_episodic_memory(
        user_id=user_id,
        conversation_id=conversation_id,
        summary=summary,
        embedding=embedding or [],
        metadata={"conversation_id": conversation_id, "message_count": len(messages)},
    )
    logger.info("summarize_episode: stored summary for conversation %s", conversation_id)


# ── Memory Consolidation ──────────────────────────────────────────────────────

@celery.task(
    name="app.workers.tasks.consolidate_memory",
    queue="memory",
)
def consolidate_memory(user_id: str) -> dict[str, Any]:
    """
    Re-summarise old episodic memories into higher-level abstractions.
    Scheduled weekly per user. Phase 2 — stub for now.
    """
    logger.info("consolidate_memory: user=%s (scheduled weekly)", user_id)
    return {"status": "ok", "note": "consolidation not yet implemented"}


# ── Checkpoint Cleanup ─────────────────────────────────────────────────────────

@celery.task(
    name="app.workers.tasks.cleanup_checkpoints",
    bind=True,
    max_retries=1,
    queue="memory",
)
def cleanup_checkpoints(self, older_than_days: int = 30) -> dict[str, Any]:
    """
    Delete LangGraph checkpoints for conversations that already have
    an episodic summary, older than older_than_days. Runs daily.
    """
    try:
        deleted = _run(_cleanup_checkpoints_async(older_than_days))
        return {"status": "ok", "deleted_threads": deleted}
    except Exception as exc:
        logger.error("cleanup_checkpoints failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


async def _cleanup_checkpoints_async(older_than_days: int) -> int:
    from app.core.db import get_db

    async with get_db() as conn:
        async with conn.cursor() as cur:
            # Find thread_ids that have episodic summaries and are old enough
            await cur.execute(
                """
                SELECT DISTINCT c.thread_id
                FROM   conversations c
                JOIN   episodic_memory em ON em.conversation_id = c.id
                WHERE  c.ended_at < now() - (%s || ' days')::interval
                """,
                (str(older_than_days),),
            )
            rows = await cur.fetchall()

    if not rows:
        return 0

    # Delete from LangGraph checkpoint tables
    thread_ids = [r["thread_id"] for r in rows]
    from app.core.db import get_db

    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                [(t,) for t in thread_ids],
            )
            await cur.executemany(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                [(t,) for t in thread_ids],
            )
            await cur.executemany(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                [(t,) for t in thread_ids],
            )

    logger.info("cleanup_checkpoints: removed %d threads", len(thread_ids))
    return len(thread_ids)
