"""
Admin metrics endpoint.

GET /api/admin/metrics — returns cost and token usage for the authenticated user.

In production, scope this to admin/enterprise plans via:
    if user.plan not in ("pro", "enterprise"):
        raise HTTPException(403, ...)
For now, every authenticated user can query their own metrics.
"""

from fastapi import APIRouter, Depends

from app.auth.deps import get_current_user
from app.auth.models import CurrentUser
from app.core.db import get_db
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/metrics")
async def get_metrics(user: CurrentUser = Depends(get_current_user)):
    """
    Return cost and usage metrics for the current user.

    Response shape:
        budget:         token budget and utilisation
        daily_costs:    last 30 days of spend grouped by day
        model_breakdown: cost per model (all time)
        recent_conversations: last 10 conversations with token counts
    """
    async with get_db() as conn:
        async with conn.cursor() as cur:
            # Budget overview
            await cur.execute(
                "SELECT token_budget, tokens_used, plan FROM users WHERE id = %s",
                (user.user_id,),
            )
            user_row = await cur.fetchone()

            # Daily cost — last 30 days
            await cur.execute(
                """
                SELECT DATE(created_at) AS day,
                       SUM(cost_estimate)::float AS cost,
                       SUM(tokens_in + tokens_out) AS tokens
                FROM   cost_tracking
                WHERE  user_id = %s
                  AND  created_at >= now() - interval '30 days'
                GROUP  BY day
                ORDER  BY day DESC
                """,
                (user.user_id,),
            )
            daily_costs = await cur.fetchall()

            # Per-model breakdown (all time)
            await cur.execute(
                """
                SELECT model_name,
                       COUNT(*) AS requests,
                       SUM(tokens_in) AS tokens_in,
                       SUM(tokens_out) AS tokens_out,
                       SUM(cost_estimate)::float AS total_cost
                FROM   cost_tracking
                WHERE  user_id = %s
                GROUP  BY model_name
                ORDER  BY total_cost DESC
                """,
                (user.user_id,),
            )
            model_breakdown = await cur.fetchall()

            # Recent conversations
            await cur.execute(
                """
                SELECT id, title, message_count, total_tokens, started_at, ended_at
                FROM   conversations
                WHERE  user_id = %s
                ORDER  BY started_at DESC
                LIMIT  10
                """,
                (user.user_id,),
            )
            recent_conversations = await cur.fetchall()

    log.info("metrics_requested", user_id=user.user_id)

    return {
        "budget": {
            "plan": user_row["plan"] if user_row else user.plan,
            "token_budget": user_row["token_budget"] if user_row else 0,
            "tokens_used": user_row["tokens_used"] if user_row else 0,
            "utilization_pct": round(
                (user_row["tokens_used"] / user_row["token_budget"] * 100)
                if user_row and user_row["token_budget"] > 0
                else 0,
                1,
            ),
        },
        "daily_costs": [
            {
                "day": str(row["day"]),
                "cost_usd": round(float(row["cost"]), 6),
                "tokens": row["tokens"],
            }
            for row in daily_costs
        ],
        "model_breakdown": [dict(row) for row in model_breakdown],
        "recent_conversations": [
            {
                "id": str(row["id"]),
                "title": row["title"],
                "message_count": row["message_count"],
                "total_tokens": row["total_tokens"],
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
            }
            for row in recent_conversations
        ],
    }
