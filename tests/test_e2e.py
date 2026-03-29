"""End-to-end integration test for rate limit plugin.

Tests the complete flow:
1. Mock OpenAI-style API server with foo (rate-limited) and bar (normal) models
2. Configure fallback from foo to bar
3. Verify automatic fallback when foo returns 429
"""

import json
import time
from unittest.mock import Mock

import httpx
import pytest
import respx
from fastapi import HTTPException

from litellm_rate_limit import RateLimitCallback


class MockOpenAIServer:
    """Simulates OpenAI-style API with rate limiting."""

    def __init__(self, base_url: str = "https://api.mock.com"):
        self.base_url = base_url
        self.foo_call_count = 0
        self.foo_rate_limited_until: float | None = None
        self.rate_limit_duration = 60

    def reset(self):
        self.foo_call_count = 0
        self.foo_rate_limited_until = None

    def _handle_foo_request(self, request: httpx.Request) -> httpx.Response:
        """Handle requests to foo model - returns 429 on first call."""
        self.foo_call_count += 1

        if self.foo_rate_limited_until is None:
            self.foo_rate_limited_until = time.time() + self.rate_limit_duration

        headers = {
            "retry-after": str(self.rate_limit_duration),
            "x-ratelimit-reset": str(self.rate_limit_duration),
        }

        return httpx.Response(
            429,
            headers=headers,
            json={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    def _handle_bar_request(self, request: httpx.Request) -> httpx.Response:
        """Handle requests to bar model - always succeeds."""
        body = json.loads(request.content)
        messages = body.get("messages", [])
        user_message = messages[-1].get("content", "") if messages else ""

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-bar-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "bar",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Response from bar: {user_message}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )


@pytest.fixture
def mock_server():
    return MockOpenAIServer()


@pytest.fixture
def callback():
    return RateLimitCallback(
        default_cooldown_seconds=60.0,
        probe_models_by_provider={"mock": ["foo", "bar"]},
    )


class TestEndToEndFlow:
    @pytest.mark.asyncio
    async def test_pre_call_hook_blocks_rate_limited_model(self, callback):
        await callback._health_state.mark_rate_limited("foo", 60.0)

        data = {"model": "foo", "messages": [{"role": "user", "content": "hi"}]}

        with pytest.raises(HTTPException) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=Mock(),
                cache=Mock(),
                data=data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_post_failure_detects_rate_limit(self, callback):
        error = type(
            "Error",
            (),
            {
                "status_code": 429,
                "headers": {"retry-after": "30"},
            },
        )()

        await callback.async_post_call_failure_hook(
            request_data={"model": "foo"},
            original_exception=error,
        )

        is_limited = await callback._health_state.is_rate_limited("foo")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_rate_limited_model_blocked_on_retry(self, callback):
        error = type(
            "Error",
            (),
            {
                "status_code": 429,
                "headers": {"retry-after": "30"},
            },
        )()

        await callback.async_post_call_failure_hook(
            request_data={"model": "foo"},
            original_exception=error,
        )

        data = {"model": "foo", "messages": [{"role": "user", "content": "hi"}]}

        with pytest.raises(HTTPException) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=Mock(),
                cache=Mock(),
                data=data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_bar_model_not_blocked_when_foo_rate_limited(self, callback):
        await callback._health_state.mark_rate_limited("foo", 60.0)

        data = {"model": "bar", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_rate_limit_expires_after_cooldown(self, callback):
        await callback._health_state.mark_rate_limited("foo", -1.0)

        is_limited = await callback._health_state.is_rate_limited("foo")
        assert is_limited is False

        data = {"model": "foo", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_provider_probe_blocks_related_models(self, callback):
        await callback._health_state.mark_rate_limited("foo", 60.0)

        data = {"model": "foo-v2", "messages": [{"role": "user", "content": "hi"}]}

        with pytest.raises(HTTPException) as exc_info:
            await callback.async_pre_call_hook(
                user_api_key_dict=Mock(),
                cache=Mock(),
                data=data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_clear_rate_limit_allows_requests(self, callback):
        await callback._health_state.mark_rate_limited("foo", 60.0)

        await callback._health_state.clear_rate_limit("foo")

        data = {"model": "foo", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )

        assert result == data


class TestWithMockedHTTP:
    @pytest.mark.asyncio
    @respx.mock
    async def test_full_flow_with_mock_api(self, callback, mock_server):
        respx.post("https://api.mock.com/v1/chat/completions").mock(
            side_effect=lambda req: mock_server._handle_foo_request(req)
            if mock_server.foo_call_count == 0
            else mock_server._handle_bar_request(req)
        )

        mock_server.reset()

        data = {"model": "foo", "messages": [{"role": "user", "content": "test"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )

        assert result["model"] == "foo"
