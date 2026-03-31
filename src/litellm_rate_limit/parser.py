"""Rate limit header parsing utilities.

Ported from LazyRouter retry_handler.py to extract reset times from API headers.
Supports:
- anthropic-ratelimit-unified-reset (Unix timestamp)
- retry-after (HTTP-date or seconds)
- x-ratelimit-reset (seconds)
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

DEFAULT_COOLDOWN_SECONDS = 60.0


def detect_api_error(result: Any) -> tuple[bool, int, str] | None:
    if result is None:
        return None

    status_code = getattr(result, "status_code", None)
    code = getattr(result, "code", None)
    if status_code is None and code is not None:
        status_code = code
    if status_code is not None:
        try:
            status_int = int(status_code)
        except (ValueError, TypeError):
            status_int = None
        if status_int is not None and status_int >= 400:
            message = getattr(result, "message", None)
            if message is None:
                message = f"HTTP {status_int}"
            return (True, status_int, str(message))
        return None

    error_response = getattr(result, "response", None)
    if error_response is not None:
        resp_status = getattr(error_response, "status_code", None)
        if resp_status is not None:
            try:
                resp_int = int(resp_status)
            except (ValueError, TypeError):
                resp_int = None
            if resp_int is not None and resp_int >= 400:
                resp_text = getattr(error_response, "text", str(error_response))
                return (True, resp_int, str(resp_text))

    error_field = getattr(result, "error", None)
    if error_field is not None:
        if isinstance(error_field, dict):
            message = error_field.get("message", str(error_field))
            raw_code = error_field.get("code", 400)
        else:
            message = str(error_field)
            raw_code = 400
        try:
            status_int = int(raw_code)
        except (ValueError, TypeError):
            status_int = 400
        return (True, status_int, message)

    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        if isinstance(error, dict):
            message = error.get("message", str(error))
            raw_code = error.get("code", error.get("status_code", 400))
        else:
            message = str(error)
            raw_code = 400
        try:
            status_int = int(raw_code)
        except (ValueError, TypeError):
            status_int = 400
        return (True, status_int, message)

    return None


def is_rate_limit_error(error: Exception) -> bool:
    detected = detect_api_error(error)
    if detected is not None:
        is_err, code, _ = detected
        return is_err and code == 429
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


def _parse_anthropic_reset_header(value: str) -> datetime | None:
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


def _parse_retry_after_header(value: str) -> datetime | None:
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


def _parse_x_ratelimit_reset_header(value: str) -> datetime | None:
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


def extract_rate_limit_reset_dt(error: Exception) -> datetime | None:
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
