"""Unit tests for rate limit header parsing."""

from datetime import datetime, timedelta, timezone
from email.utils import formatdate
from typing import Any

import pytest
from freezegun import freeze_time

from litellm_rate_limit.parser import (
    DEFAULT_COOLDOWN_SECONDS,
    extract_rate_limit_reset_dt,
    extract_rate_limit_reset_seconds,
    is_rate_limit_error,
)


class MockResponse:
    def __init__(self, headers: dict[str, Any]):
        self.headers = headers


class MockError:
    def __init__(self, headers: dict[str, Any] = None, response_headers: dict[str, Any] = None):
        self.headers = headers or {}
        if response_headers is not None:
            self.response = MockResponse(response_headers)
        self._response = None


class MockAnthropicError:
    def __init__(self, headers: dict[str, Any]):
        self._response = MockResponse(headers)
        self.headers = {}


class TestIsRateLimitError:
    def test_is_rate_limit_error_429_status_code(self):
        error = type("Error", (), {"status_code": 429})()
        assert is_rate_limit_error(error) is True

    def test_is_rate_limit_error_429_code(self):
        error = type("Error", (), {"status_code": None, "code": 429})()
        assert is_rate_limit_error(error) is True

    def test_is_rate_limit_error_response_status_code(self):
        response = type("Response", (), {"status_code": 429})()
        error = type("Error", (), {"response": response})()
        assert is_rate_limit_error(error) is True

    def test_is_rate_limit_error_other_codes(self):
        error = type("Error", (), {"status_code": 500})()
        assert is_rate_limit_error(error) is False

    def test_is_rate_limit_error_no_status(self):
        error = Exception("Some error")
        assert is_rate_limit_error(error) is False


class TestExtractAnthropicResetHeader:
    def test_extract_anthropic_reset_header(self):
        future_dt = datetime.now(timezone.utc) + timedelta(seconds=120)
        future_timestamp = str(future_dt.timestamp())

        error = MockAnthropicError({"anthropic-ratelimit-unified-reset": future_timestamp})

        result = extract_rate_limit_reset_dt(error)
        assert result is not None
        assert abs((result - future_dt).total_seconds()) < 1

    def test_extract_anthropic_reset_header_invalid(self):
        error = MockAnthropicError({"anthropic-ratelimit-unified-reset": "invalid"})

        result = extract_rate_limit_reset_dt(error)
        assert result is None


class TestExtractRetryAfterHeader:
    def test_extract_retry_after_seconds(self):
        with freeze_time("2024-01-01 12:00:00", tz_offset=0):
            error = MockError(headers={"retry-after": "60"})

            result = extract_rate_limit_reset_dt(error)
            expected = datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
            assert result == expected

    def test_extract_retry_after_http_date(self):
        future_dt = datetime.now(timezone.utc) + timedelta(seconds=120)
        http_date = formatdate(future_dt.timestamp(), usegmt=True)

        error = MockError(headers={"retry-after": http_date})

        result = extract_rate_limit_reset_dt(error)
        assert result is not None
        assert abs((result - future_dt).total_seconds()) < 1

    def test_extract_retry_after_invalid(self):
        error = MockError(headers={"retry-after": "invalid"})

        result = extract_rate_limit_reset_dt(error)
        assert result is None


class TestExtractXRateLimitResetHeader:
    def test_extract_x_ratelimit_reset(self):
        with freeze_time("2024-01-01 12:00:00", tz_offset=0):
            error = MockError(response_headers={"x-ratelimit-reset": "30.5"})

            result = extract_rate_limit_reset_dt(error)
            expected = datetime(2024, 1, 1, 12, 0, 30, 500000, tzinfo=timezone.utc)
            assert result == expected


class TestExtractRateLimitResetSeconds:
    def test_extract_rate_limit_reset_seconds(self):
        with freeze_time("2024-01-01 12:00:00", tz_offset=0):
            error = MockError(headers={"retry-after": "45"})

            result = extract_rate_limit_reset_seconds(error)
            assert result == 45.0

    def test_extract_no_header_returns_default(self):
        error = Exception("No headers")
        result = extract_rate_limit_reset_seconds(error)
        assert result == DEFAULT_COOLDOWN_SECONDS

    def test_extract_negative_reset_returns_default(self):
        with freeze_time("2024-01-01 12:00:00", tz_offset=0):
            past_dt = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
            error = MockAnthropicError({"anthropic-ratelimit-unified-reset": str(past_dt.timestamp())})

            result = extract_rate_limit_reset_seconds(error)
            assert result == DEFAULT_COOLDOWN_SECONDS

    def test_extract_custom_default(self):
        error = Exception("No headers")
        result = extract_rate_limit_reset_seconds(error, default=120.0)
        assert result == 120.0


class TestHeaderPriority:
    def test_anthropic_header_takes_priority(self):
        future_dt = datetime.now(timezone.utc) + timedelta(seconds=100)
        anthropic_timestamp = str(future_dt.timestamp())

        error = MockAnthropicError(
            {
                "anthropic-ratelimit-unified-reset": anthropic_timestamp,
                "retry-after": "999",
                "x-ratelimit-reset": "888",
            }
        )

        result = extract_rate_limit_reset_dt(error)
        assert result is not None
        assert abs((result - future_dt).total_seconds()) < 1

    def test_retry_after_when_no_anthropic(self):
        with freeze_time("2024-01-01 12:00:00", tz_offset=0):
            error = MockError(
                response_headers={
                    "retry-after": "50",
                    "x-ratelimit-reset": "999",
                }
            )

            result = extract_rate_limit_reset_dt(error)
            expected = datetime(2024, 1, 1, 12, 0, 50, tzinfo=timezone.utc)
            assert result == expected


class TestCaseInsensitiveHeaders:
    def test_case_insensitive_headers(self):
        error = MockError(headers={"Retry-After": "30"})

        result = extract_rate_limit_reset_seconds(error)
        assert result == pytest.approx(30.0, abs=0.1)
