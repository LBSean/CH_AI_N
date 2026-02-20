"""
Structured logging via structlog.

All application log entries are JSON with consistent fields:
  timestamp, level, event, user_id, model, tokens_in, tokens_out,
  latency_ms, tool_name, error_type, ...

LangSmith handles LLM-level trace inspection.
structlog handles application-level events.

Usage:
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("agent_invoke", user_id=user_id, model="gpt-4o", latency_ms=1200)
"""

import logging
import sys

import structlog
from app.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structlog processors. Call once at application startup.
    Development: pretty colored output.
    Production:  JSON output (machine-readable for cloud logging).
    """
    settings = get_settings()
    is_dev = settings.environment == "development"

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if is_dev else logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
