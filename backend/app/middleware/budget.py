"""
Budget enforcement — blocks LLM calls when a user has exhausted their token budget.

Applied as a FastAPI dependency on protected routes:
    user: CurrentUser = Depends(check_budget)

At 80% utilization: logs a warning.
At 100%: raises HTTP 429.
"""

from fastapi import Depends, HTTPException, status

from app.auth.deps import get_current_user
from app.auth.models import CurrentUser
from app.auth.service import get_user_by_id
from app.core.logging import get_logger

log = get_logger(__name__)


async def check_budget(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """
    FastAPI dependency. Returns the user if under budget, raises 429 if exceeded.
    Logs a warning when utilization crosses 80%.
    """
    user_data = await get_user_by_id(user.user_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    budget: int = user_data["token_budget"]
    used: int = user_data["tokens_used"]

    if budget > 0:
        utilization = used / budget
        if utilization >= 1.0:
            log.warning(
                "budget_exceeded",
                user_id=user.user_id,
                tokens_used=used,
                token_budget=budget,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Token budget exhausted ({used}/{budget}). "
                       "Upgrade your plan or wait for the next billing cycle.",
            )
        if utilization >= 0.8:
            log.warning(
                "budget_warning",
                user_id=user.user_id,
                utilization=round(utilization * 100, 1),
            )

    return user
