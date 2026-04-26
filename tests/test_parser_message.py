"""Test parsing reset time from error messages."""

from datetime import timezone

from litellm_rate_limit.parser import (
    _extract_reset_time_from_message,
    extract_rate_limit_reset_seconds,
)


class MockError:
    def __init__(self, message):
        self.message = message


class TestExtractResetTimeFromMessage:
    def test_zai_format(self):
        error = MockError("Usage limit reached for 5 hour. Your limit will reset at 2026-04-25 18:48:34")
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 25
        assert result.hour == 18
        assert result.minute == 48
        assert result.second == 34
        assert result.tzinfo == timezone.utc

    def test_rfc2822_format(self):
        error = MockError("Rate limit exceeded. Reset at Mon, 25 Apr 2026 18:48:34 GMT")
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.year == 2026
        assert result.tzinfo == timezone.utc

    def test_no_reset_time(self):
        error = MockError("Some random error message without reset time")
        result = _extract_reset_time_from_message(error)

        assert result is None

    def test_extract_seconds(self):
        error = MockError("Usage limit reached for 5 hour. Your limit will reset at 2026-04-25 18:48:34")
        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        # Should parse the reset time, not use default
        assert seconds != 3600
        assert seconds > 0

    def test_naive_datetime_has_utc(self):
        error = MockError("reset at 2026-04-25 18:48:34")
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.tzinfo == timezone.utc
