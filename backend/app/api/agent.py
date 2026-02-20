"""
Agent API endpoints.

All routes require JWT auth + rate limiting + budget check.
Conversation and message records are written asynchronously via BackgroundTasks.
"""

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agents.research_agent import build_research_graph
from app.auth.models import CurrentUser
from app.middleware.budget import check_budget
from app.middleware.rate_limit import check_rate_limit
from app.middleware.sanitize import sanitize_message
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

# Cost rates (USD per 1K tokens) — update as provider pricing changes
_COST_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o":            (0.005, 0.015),
    "gpt-4o-mini":       (0.00015, 0.0006),
    "claude-sonnet-4-6": (0.003, 0.015),
}


class InvokeRequest(BaseModel):
    message: str
    thread_id: str | None = None


# ── Background helpers ─────────────────────────────────────────────────────────

async def _ensure_conversation(user_id: str, thread_id: str) -> str:
    """
    Get or create a conversation row for this thread.
    Returns the conversation UUID string.
    """
    from app.core.db import get_db
    async with get_db() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM conversations WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id),
            )
            row = await cur.fetchone()
            if row:
                return str(row["id"])

            await cur.execute(
                "INSERT INTO conversations (user_id, thread_id) VALUES (%s, %s) RETURNING id",
                (user_id, thread_id),
            )
            new_row = await cur.fetchone()
            return str(new_row["id"])


async def _log_messages(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Write user + assistant messages and update conversation stats."""
    settings = get_settings()
    in_rate, out_rate = _COST_RATES.get(model, (0.005, 0.015))
    cost = (tokens_in / 1000 * in_rate) + (tokens_out / 1000 * out_rate)

    from app.core.db import get_db
    from app.auth.service import increment_tokens_used

    async with get_db() as conn:
        async with conn.cursor() as cur:
            # Write user message
            await cur.execute(
                "INSERT INTO messages (conversation_id, user_id, role, content, token_count) "
                "VALUES (%s, %s, 'user', %s, %s)",
                (conversation_id, user_id, user_message, tokens_in),
            )
            # Write assistant message
            await cur.execute(
                "INSERT INTO messages (conversation_id, user_id, role, content, token_count, model_used) "
                "VALUES (%s, %s, 'assistant', %s, %s, %s)",
                (conversation_id, user_id, assistant_message, tokens_out, model),
            )
            # Update conversation counters
            await cur.execute(
                "UPDATE conversations "
                "SET message_count = message_count + 2, total_tokens = total_tokens + %s "
                "WHERE id = %s",
                (tokens_in + tokens_out, conversation_id),
            )
            # Write cost tracking
            await cur.execute(
                "INSERT INTO cost_tracking "
                "(user_id, conversation_id, model_name, tokens_in, tokens_out, cost_estimate) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, conversation_id, model, tokens_in, tokens_out, cost),
            )

    await increment_tokens_used(user_id, tokens_in + tokens_out)

    log.info(
        "response_logged",
        user_id=user_id,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=round(cost, 6),
    )


def _extract_usage(ai_message) -> tuple[int, int]:
    """Extract token counts from an AIMessage if usage_metadata is present."""
    meta = getattr(ai_message, "usage_metadata", None)
    if meta:
        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)
    return 0, 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/invoke")
async def invoke_agent(
    req: InvokeRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(check_rate_limit),
    _budget: CurrentUser = Depends(check_budget),
):
    """Synchronous invocation — waits for the full agent response."""
    settings = get_settings()
    sanitize_message(req.message, user.user_id)

    graph = build_research_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=req.message)],
            "user_id": user.user_id,
            "thread_id": thread_id,
            "context": "",
            "episodic_context": "",
            "tool_calls": [],
            "tool_results": {},
            "metadata": {"model": settings.primary_model},
        },
        config=config,
    )

    last_message = result["messages"][-1]
    tokens_in, tokens_out = _extract_usage(last_message)
    response_text = last_message.content

    log.info("invoke_complete", user_id=user.user_id, thread_id=thread_id)

    # Async logging — does not block the response
    background_tasks.add_task(
        _log_conversation,
        user_id=user.user_id,
        thread_id=thread_id,
        user_message=req.message,
        assistant_message=response_text,
        model=settings.primary_model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    return {"thread_id": thread_id, "response": response_text}


@router.post("/stream")
async def stream_agent(
    req: InvokeRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(check_rate_limit),
    _budget: CurrentUser = Depends(check_budget),
):
    """
    Streaming invocation via Server-Sent Events (SSE).

    Events:
      {"type": "thread_id", "thread_id": "..."}  — first event
      {"type": "token",     "content": "..."}     — streamed tokens
      {"type": "done"}                            — end of stream
    """
    settings = get_settings()
    sanitize_message(req.message, user.user_id)

    graph = build_research_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        full_response: list[str] = []
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

        async for event in graph.astream_events(
            {
                "messages": [HumanMessage(content=req.message)],
                "user_id": user.user_id,
                "thread_id": thread_id,
                "context": "",
                "episodic_context": "",
                "tool_calls": [],
                "tool_results": {},
                "metadata": {"model": settings.primary_model},
            },
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_response.append(chunk.content)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # Fire-and-forget: log after streaming completes
        background_tasks.add_task(
            _log_conversation,
            user_id=user.user_id,
            thread_id=thread_id,
            user_message=req.message,
            assistant_message="".join(full_response),
            model=settings.primary_model,
            tokens_in=0,
            tokens_out=0,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/threads/{thread_id}")
async def get_thread_state(
    thread_id: str,
    user: CurrentUser = Depends(check_rate_limit),
):
    """Retrieve the persisted state of a conversation thread."""
    graph = build_research_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if not state:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "state": state.values}


# ── Composite background helper ────────────────────────────────────────────────

async def _log_conversation(
    user_id: str,
    thread_id: str,
    user_message: str,
    assistant_message: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Ensure conversation exists, then log messages + cost."""
    try:
        conversation_id = await _ensure_conversation(user_id, thread_id)
        await _log_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    except Exception as exc:
        log.error("log_conversation_failed", user_id=user_id, error=str(exc))
