"""Rate limit header parsing utilities.

Ported from LazyRouter retry_handler.py to extract reset times from API headers.
Supports:
- anthropic-ratelimit-unified-reset (Unix timestamp)
- retry-after (HTTP-date or seconds)
- x-ratelimit-reset (seconds)
"""

import contextlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

DEFAULT_COOLDOWN_SECONDS = 60.0

# Cache local timezone to avoid repeated system calls
_local_timezone = None


def _get_local_timezone() -> timezone:
    """Get the local system timezone."""
    global _local_timezone
    if _local_timezone is None:
        _local_timezone = datetime.now().astimezone().tzinfo
    return _local_timezone


def detect_api_error(result: Any) -> tuple[bool, int, str] | None:
    if result is None:
        return None

    # Check for LiteLLM RouterRateLimitError first (has .status_code but not as int)
    if hasattr(result, "status_code"):
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
            # Handle RouterRateLimitError which has status_code as a string or code attribute
            if status_int is None and status_code in (429, "429"):
                message = getattr(result, "message", "Rate limit exceeded")
                return (True, 429, str(message))

    # Check for .code attribute (some LiteLLM exceptions use this instead)
    code = getattr(result, "code", None)
    if code is not None:
        try:
            status_int = int(code)
        except (ValueError, TypeError):
            status_int = None
        if status_int is not None and status_int >= 400:
            message = getattr(result, "message", None)
            if message is None:
                message = f"HTTP {status_int}"
            return (True, status_int, str(message))

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

    text = str(result)
    match = re.search(r"Error code:\s*(\d{3})", text)
    if match:
        status_int = int(match.group(1))
        if status_int >= 400:
            return (True, status_int, text)

    _known_error_status = {
        "BadRequestError": 400,
        "UnauthorizedError": 401,
        "AuthenticationError": 401,
        "PermissionDeniedError": 403,
        "ForbiddenError": 403,
        "NotFoundError": 404,
        "ConflictError": 409,
        "UnprocessableEntityError": 422,
        "RateLimitError": 429,
        "InternalServerError": 500,
        "ServiceUnavailableError": 503,
        "APIConnectionError": 502,
        "APITimeoutError": 504,
    }
    cls_name = type(result).__name__
    if cls_name in _known_error_status:
        return (True, _known_error_status[cls_name], text)

    return None


def is_rate_limit_error(error: Exception) -> bool:
    detected = detect_api_error(error)
    if detected is not None:
        is_err, code, _ = detected
        return is_err and code == 429
    return False


def _get_headers_from_error(error: Exception) -> dict:
    headers = {}

    try:
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            headers = dict(error.response.headers)
        elif hasattr(error, "headers") and error.headers:
            headers = dict(error.headers)
        elif hasattr(error, "_response") and hasattr(error._response, "headers"):
            headers = dict(error._response.headers)
    except (TypeError, AttributeError):
        pass

    return {k.lower(): v for k, v in headers.items()}


def _extract_reset_time_from_message(error: Exception) -> datetime | None:
    """Extract reset time from error message body.

    Handles non-standard error formats like Zai's:
    "Usage limit reached for 5 hour. Your limit will reset at 2026-04-25 18:48:34"

    Args:
        error: Exception that may contain reset time in message.

    Returns:
        datetime of reset time, or None if not found.
    """
    from email.utils import parsedate_to_datetime

    # Try to get message from exception
    message = None

    # Try different attribute patterns
    if hasattr(error, "message") and error.message:
        message = str(error.message)
    elif hasattr(error, "response"):
        response = error.response
        if hasattr(response, "text") and response.text:
            message = response.text
        elif hasattr(response, "json"):
            with contextlib.suppress(BaseException):
                message = response.json().get("error", {}).get("message")

    if not message and isinstance(error, dict):
        message = (
            error.get("error", {}).get("message") if isinstance(error.get("error"), dict) else str(error)
        )

    if not message:
        return None

    # Look for patterns such as "reset at 2026-04-25 18:48:34" or "reset at Mon, 25 Apr 2026 18:48:34 GMT"

    patterns = [
        r"[Rr]eset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        r"[Rr]eset at ([A-Z][a-z]+, \d+ [A-Z][a-z]+ \d+ \d{2}:\d{2}:\d{2} [A-Z]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            date_str = match.group(1)
            # Try parsing as ISO format first
            with contextlib.suppress(BaseException):
                dt = datetime.fromisoformat(date_str.replace(" ", "T"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_get_local_timezone())
                return dt
            # Try RFC 2822 format
            with contextlib.suppress(BaseException):
                dt = parsedate_to_datetime(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_get_local_timezone())
                return dt

    return None


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

    # Try to extract from error message body (for non-standard providers like Zai)
    return _extract_reset_time_from_message(error)


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

    now = datetime.now(_get_local_timezone())
    delta = reset_dt - now
    seconds = delta.total_seconds()

    # Return default if reset time is in the past
    if seconds <= 0:
        return default

    return seconds
