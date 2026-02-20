"""
Input sanitization — lightweight prompt injection filter.

Checks user messages for common prompt injection patterns before they
enter the agent graph. Not a silver bullet — defence in depth alongside
proper system prompt separation.
"""

import re
from fastapi import HTTPException, status

from app.core.logging import get_logger

log = get_logger(__name__)

# Patterns that signal likely prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+\w+", re.I),
    re.compile(r"act\s+as\s+(?:a|an|the)\s+\w+", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
    re.compile(r"<\s*/?system\s*>", re.I),
    re.compile(r"\[INST\]", re.I),
    re.compile(r"###\s*instruction", re.I),
]

# Hard length cap — prevents token-flooding attacks
_MAX_MESSAGE_LENGTH = 8_000  # characters


def sanitize_message(text: str, user_id: str) -> str:
    """
    Validate and clean a user message.
    Raises HTTP 400 on injection detection or length violation.
    Returns the (unchanged) text if clean.
    """
    if len(text) > _MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message too long ({len(text)} chars). Maximum is {_MAX_MESSAGE_LENGTH}.",
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            log.warning(
                "prompt_injection_detected",
                user_id=user_id,
                pattern=pattern.pattern,
                message_excerpt=text[:100],
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message contains disallowed content.",
            )

    return text
