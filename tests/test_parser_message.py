"""Test parsing reset time from error messages."""

from datetime import datetime, timedelta

from litellm_rate_limit.parser import (
    _extract_reset_time_from_message,
    extract_rate_limit_reset_seconds,
)


class MockError:
    def __init__(self, message):
        self.message = message


def make_future_message(template: str, hours_ahead: int = 24) -> tuple[str, datetime]:
    """Generate a message with a future datetime and return both the message and expected datetime."""
    local_tz = datetime.now().astimezone().tzinfo
    future_dt = datetime.now(local_tz) + timedelta(hours=hours_ahead)
    date_str = future_dt.strftime("%Y-%m-%d %H:%M:%S")
    return template.format(date_str=date_str), future_dt


class TestExtractResetTimeFromMessage:
    def test_zai_format(self):
        msg, expected_dt = make_future_message(
            "Usage limit reached for 5 hour. Your limit will reset at {date_str}"
        )
        error = MockError(msg)
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.year == expected_dt.year
        assert result.month == expected_dt.month
        assert result.day == expected_dt.day
        assert result.hour == expected_dt.hour
        assert result.minute == expected_dt.minute
        assert result.second == expected_dt.second
        assert result.tzinfo is not None

    def test_rfc2822_format(self):
        local_tz = datetime.now().astimezone().tzinfo
        future_dt = datetime.now(local_tz) + timedelta(hours=24)
        date_str = future_dt.strftime("%d %b %Y %H:%M:%S")
        msg = f"Rate limit exceeded. Reset at Mon, {date_str} GMT"
        error = MockError(msg)
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.year == future_dt.year
        assert result.tzinfo is not None

    def test_no_reset_time(self):
        error = MockError("Some random error message without reset time")
        result = _extract_reset_time_from_message(error)

        assert result is None

    def test_extract_seconds(self):
        """Test that extract_rate_limit_reset_seconds returns parsed value, not default."""
        msg, _ = make_future_message(
            "Usage limit reached for 5 hour. Your limit will reset at {date_str}", hours_ahead=2
        )
        error = MockError(msg)
        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        assert seconds != 3600
        assert 7000 < seconds < 7500

    def test_naive_datetime_has_local_tz(self):
        msg, _ = make_future_message("reset at {date_str}")
        error = MockError(msg)
        result = _extract_reset_time_from_message(error)

        assert result is not None
        assert result.tzinfo is not None
