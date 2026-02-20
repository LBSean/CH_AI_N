"""
Per-user rate limiting via Redis sliding window.

Window: 60 seconds (configurable via settings.rate_limit_window).
Limit:  users.rate_limit requests per window (plan-based, stored in Postgres).

Applied as a FastAPI dependency:
    user: CurrentUser = Depends(check_rate_limit)
"""

import time

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from functools import lru_cache

from app.auth.deps import get_current_user
from app.auth.models import CurrentUser
from app.auth.service import get_user_by_id
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_WINDOW_SECONDS = 60


@lru_cache
def _get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def check_rate_limit(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """
    FastAPI dependency. Enforces per-user rate limit using a Redis sorted set
    (sliding window algorithm). Raises 429 with Retry-After header on breach.
    """
    user_data = await get_user_by_id(user.user_id)
    limit: int = user_data["rate_limit"] if user_data else 60

    redis = _get_redis()
    key = f"ratelimit:{user.user_id}"
    now = time.time()
    window_start = now - _WINDOW_SECONDS

    async with redis.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, 0, window_start)   # evict old entries
        pipe.zadd(key, {str(now): now})                # add this request
        pipe.zcard(key)                                # count requests in window
        pipe.expire(key, _WINDOW_SECONDS + 1)          # TTL cleanup
        results = await pipe.execute()

    count: int = results[2]

    if count > limit:
        log.warning("rate_limit_exceeded", user_id=user.user_id, count=count, limit=limit)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({count}/{limit} req/min).",
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )

    return user
