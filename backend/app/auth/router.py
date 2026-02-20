"""Auth endpoints: register, login, refresh."""

import psycopg.errors
from fastapi import APIRouter, HTTPException, status
from jose import JWTError

from app.auth.models import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse
from app.auth.security import create_access_token, create_refresh_token, decode_token
from app.auth.service import authenticate_user, create_user, get_user_by_id

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    try:
        user = await create_user(req.email, req.password)
    except Exception as exc:
        # Catches UniqueViolation from psycopg
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=500, detail="Registration failed")

    return TokenResponse(
        access_token=create_access_token(str(user["id"]), user["email"], user["plan"]),
        refresh_token=create_refresh_token(str(user["id"])),
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = await authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return TokenResponse(
        access_token=create_access_token(str(user["id"]), user["email"], user["plan"]),
        refresh_token=create_refresh_token(str(user["id"])),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    try:
        payload = decode_token(req.refresh_token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    user = await get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=create_access_token(str(user["id"]), user["email"], user["plan"]),
        refresh_token=create_refresh_token(str(user["id"])),
    )
