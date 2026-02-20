from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from app.auth.deps import get_current_user
from app.auth.models import CurrentUser
from app.workers.tasks import ingest_document

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/file")
async def ingest_file(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Accept a file upload and queue it for background ingestion.
    Returns immediately with a task ID — ingestion runs in the Celery worker.
    """
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    task = ingest_document.delay(
        text=text,
        metadata={"filename": file.filename},
        user_id=user.user_id,
    )
    return {"status": "queued", "task_id": task.id}


class TextIngestRequest(BaseModel):
    text: str
    metadata: dict = {}


@router.post("/text")
async def ingest_text(
    req: TextIngestRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Queue raw text for background ingestion.
    Returns immediately with a task ID.
    """
    task = ingest_document.delay(
        text=req.text,
        metadata=req.metadata,
        user_id=user.user_id,
    )
    return {"status": "queued", "task_id": task.id}
