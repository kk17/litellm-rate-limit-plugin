"""Test parsing reset time from error messages with realistic error objects."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from litellm_rate_limit.parser import extract_rate_limit_reset_seconds


def make_future_message(hours_ahead: int = 24) -> str:
    """Generate an error message with a future datetime."""
    local_tz = datetime.now().astimezone().tzinfo
    future_dt = datetime.now(local_tz) + timedelta(hours=hours_ahead)
    return f"Usage limit reached. Your limit will reset at {future_dt.strftime('%Y-%m-%d %H:%M:%S')}"


class TestExtractRateLimitResetSecondsWithRealErrors:
    def test_zai_error_with_message_attribute(self):
        """Test with error that has .message attribute like Zai's OpenAIError."""
        future_msg = make_future_message(hours_ahead=2)
        error = Mock()
        error.message = f"Error code: 429 - {{'error': {{'code': '1308', 'message': '{future_msg}'}}}}"

        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        assert seconds != 3600
        assert 7000 < seconds < 7500

    def test_zai_error_with_response_text(self):
        """Test with error that has .response.text attribute."""
        error = Mock()
        error.message = None
        error.response = Mock()
        error.response.text = make_future_message(hours_ahead=2)

        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        assert seconds != 3600
        assert 7000 < seconds < 7500

    def test_no_offset_naive_datetime_error(self):
        """Ensure no TypeError when subtracting datetimes with different timezone awareness."""
        error = Mock()
        error.message = make_future_message(hours_ahead=1)

        try:
            seconds = extract_rate_limit_reset_seconds(error, default=3600)
            assert seconds > 0
        except TypeError as e:
            pytest.fail(f"Should not raise TypeError: {e}")
