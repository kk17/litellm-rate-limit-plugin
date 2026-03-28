"""Rate limit header parsing utilities.

Ported from LazyRouter retry_handler.py to extract reset times from API headers.
Supports:
- anthropic-ratelimit-unified-reset (Unix timestamp)
- retry-after (HTTP-date or seconds)
- x-ratelimit-reset (seconds)
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

DEFAULT_COOLDOWN_SECONDS = 60.0


def is_rate_limit_error(error: Exception) -> bool:
    """Detect if an exception is a 429 rate limit error.

    Args:
        error: Exception to check.

    Returns:
        True if the error is a rate limit (429) error.
    """
    # Check for common rate limit exception patterns
    if hasattr(error, "status_code") and error.status_code == 429:
        return True
    if hasattr(error, "code") and error.code == 429:
        return True
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code == 429
    # Check litellm exceptions
    if hasattr(error, "llm_provider") and "429" in str(error):
        return True
    return False


def _get_headers_from_error(error: Exception) -> dict:
    """Extract headers from an exception.

    Args:
        error: Exception that may contain response headers.

    Returns:
        Dictionary of headers (lowercase keys).
    """
    headers = {}

    # Try litellm exception pattern
    if hasattr(error, "response") and hasattr(error.response, "headers"):
        headers = dict(error.response.headers)
    # Try openai exception pattern
    elif hasattr(error, "headers") and error.headers:
        headers = dict(error.headers)
    # Try anthropic exception pattern
    elif hasattr(error, "_response") and hasattr(error._response, "headers"):
        headers = dict(error._response.headers)

    # Normalize header keys to lowercase
    return {k.lower(): v for k, v in headers.items()}


def _parse_anthropic_reset_header(value: str) -> Optional[datetime]:
    """Parse Anthropic's ratelimit-unified-reset header.

    Format: Unix timestamp (seconds since epoch).

    Args:
        value: Header value string.

    Returns:
        datetime of reset time, or None if parsing fails.
    """
    try:
        timestamp = float(value)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_retry_after_header(value: str) -> Optional[datetime]:
    """Parse retry-after header.

    Format can be:
    - Seconds (integer)
    - HTTP-date (RFC 2822)

    Args:
        value: Header value string.

    Returns:
        datetime of reset time, or None if parsing fails.
    """
    # Try parsing as seconds first
    try:
        seconds = int(value)
        return datetime.now(timezone.utc) + __import__("datetime").timedelta(seconds=seconds)
    except ValueError:
        pass

    # Try parsing as HTTP-date
    try:
        parsed_date = parsedate_to_datetime(value)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        return parsed_date
    except (ValueError, TypeError):
        return None


def _parse_x_ratelimit_reset_header(value: str) -> Optional[datetime]:
    """Parse x-ratelimit-reset header.

    Format: Seconds until reset.

    Args:
        value: Header value string.

    Returns:
        datetime of reset time, or None if parsing fails.
    """
    try:
        seconds = float(value)
        return datetime.now(timezone.utc) + __import__("datetime").timedelta(seconds=seconds)
    except (ValueError, TypeError):
        return None


def extract_rate_limit_reset_dt(error: Exception) -> Optional[datetime]:
    """Parse reset time from response headers.

    Checks headers in order of preference:
    1. anthropic-ratelimit-unified-reset (exact timestamp)
    2. retry-after (HTTP-date or seconds)
    3. x-ratelimit-reset (seconds)

    Args:
        error: Exception containing response headers.

    Returns:
        datetime of when rate limit resets, or None if not found.
    """
    headers = _get_headers_from_error(error)

    # Try Anthropic header first (most precise)
    if "anthropic-ratelimit-unified-reset" in headers:
        reset_dt = _parse_anthropic_reset_header(headers["anthropic-ratelimit-unified-reset"])
        if reset_dt:
            return reset_dt

    # Try retry-after header
    if "retry-after" in headers:
        reset_dt = _parse_retry_after_header(headers["retry-after"])
        if reset_dt:
            return reset_dt

    # Try x-ratelimit-reset header
    if "x-ratelimit-reset" in headers:
        reset_dt = _parse_x_ratelimit_reset_header(headers["x-ratelimit-reset"])
        if reset_dt:
            return reset_dt

    return None


def extract_rate_limit_reset_seconds(error: Exception, default: float = DEFAULT_COOLDOWN_SECONDS) -> float:
    """Calculate seconds until rate limit resets.

    Args:
        error: Exception containing response headers.
        default: Default seconds to return if no reset time found.

    Returns:
        Seconds until reset, with fallback to default.
    """
    reset_dt = extract_rate_limit_reset_dt(error)

    if reset_dt is None:
        return default

    now = datetime.now(timezone.utc)
    delta = reset_dt - now
    seconds = delta.total_seconds()

    # Return default if reset time is in the past
    if seconds <= 0:
        return default

    return seconds
