"""
FastAPI dependency for JWT-protected routes.

Usage:
    @router.post("/api/agent/invoke")
    async def invoke(req: InvokeRequest, user: CurrentUser = Depends(get_current_user)):
        ...  # user.user_id, user.email, user.plan are available
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.auth.models import CurrentUser
from app.auth.security import decode_token

_bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    """
    Verifies the Bearer JWT and returns the decoded user claims.
    Raises 401 if the token is missing, expired, or invalid.
    """
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not an access token",
        )

    return CurrentUser(
        user_id=payload["sub"],
        email=payload["email"],
        plan=payload["plan"],
    )
