"""Test parsing reset time from error messages with realistic error objects."""

from unittest.mock import Mock

import pytest

from litellm_rate_limit.parser import extract_rate_limit_reset_seconds


class TestExtractRateLimitResetSecondsWithRealErrors:
    def test_zai_error_with_message_attribute(self):
        """Test with error that has .message attribute like Zai's OpenAIError."""
        error = Mock()
        error.message = "Error code: 429 - {'error': {'code': '1308', 'message': 'Usage limit reached for 5 hour. Your limit will reset at 2026-04-25 18:48:34'}}"

        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        assert seconds != 3600
        assert seconds > 0

    def test_zai_error_with_response_text(self):
        """Test with error that has .response.text attribute."""
        error = Mock()
        error.message = None
        error.response = Mock()
        error.response.text = "Usage limit reached. Your limit will reset at 2026-04-25 18:48:34"

        seconds = extract_rate_limit_reset_seconds(error, default=3600)

        assert seconds != 3600
        assert seconds > 0

    def test_no_offset_naive_datetime_error(self):
        """Ensure no TypeError when subtracting datetimes with different timezone awareness."""
        error = Mock()
        error.message = "Usage limit reached for 5 hour. Your limit will reset at 2026-04-25 18:48:34"

        # This should not raise TypeError: can't subtract offset-naive and offset-aware datetimes
        try:
            seconds = extract_rate_limit_reset_seconds(error, default=3600)
            assert seconds > 0
        except TypeError as e:
            pytest.fail(f"Should not raise TypeError: {e}")
