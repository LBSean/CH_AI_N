from fastapi import APIRouter

from app.core.checkpointer import get_connection_pool

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Health check endpoint.  Verifies the Postgres connection is reachable."""
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {"status": "ok", "database": db_status}
