"""
Celery application instance.

Imported by task modules and by the worker process:
    celery -A app.workers.celery_app worker --queues ingestion,memory -l info
"""

import os
from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery = Celery(
    "ai_workspace",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Named queues — each with distinct priority / concurrency in production
    task_routes={
        "app.workers.tasks.ingest_document":     {"queue": "ingestion"},
        "app.workers.tasks.summarize_episode":   {"queue": "memory"},
        "app.workers.tasks.consolidate_memory":  {"queue": "memory"},
        "app.workers.tasks.cleanup_checkpoints": {"queue": "memory"},
    },
    # Retry policy defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
