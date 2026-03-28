"""Unit tests for RateLimitCallback."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from litellm_rate_limit.callback import RateLimitCallback
from litellm_rate_limit.parser import DEFAULT_COOLDOWN_SECONDS


class TestRateLimitCallback:
    def test_init_default_values(self):
        callback = RateLimitCallback()
        assert callback.default_cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
        assert callback._router is None

    def test_init_custom_values(self):
        callback = RateLimitCallback(default_cooldown_seconds=120.0)
        assert callback.default_cooldown_seconds == 120.0

    def test_set_router(self):
        callback = RateLimitCallback()
        router = Mock()
        callback.set_router(router)
        assert callback._router == router

    @pytest.mark.asyncio
    async def test_post_call_failure_rate_limit(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        error = type("Error", (), {"status_code": 429, "headers": {"retry-after": "45"}})()

        request_data = {"model": "claude-3-sonnet"}

        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )

    @pytest.mark.asyncio
    async def test_post_call_failure_other_error(self):
        callback = RateLimitCallback()

        error = Mock()
        error.status_code = 500

        request_data = {"model": "claude-3-sonnet"}

        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )

    @pytest.mark.asyncio
    async def test_update_cooldown_with_router(self):
        cooldown_cache = MagicMock()
        cooldown_cache.set_cooldown = AsyncMock()

        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.get_deployment = Mock(return_value=None)

        callback = RateLimitCallback(router=router)

        await callback._update_cooldown("claude-3-sonnet", 45.0)

        cooldown_cache.set_cooldown.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_cooldown_no_router(self):
        callback = RateLimitCallback()

        await callback._update_cooldown("claude-3-sonnet", 45.0)

    @pytest.mark.asyncio
    async def test_update_cooldown_no_cache(self):
        router = Mock(spec=["cooldown_cache"])
        del router.cooldown_cache

        callback = RateLimitCallback(router=router)

        await callback._update_cooldown("claude-3-sonnet", 45.0)

    def test_get_deployment_for_model_no_router(self):
        callback = RateLimitCallback()
        result = callback._get_deployment_for_model("claude-3-sonnet")
        assert result is None

    def test_get_deployment_for_model_with_router(self):
        router = Mock()
        deployment = Mock()
        deployment.model_info = {"id": "deployment-123"}
        router.get_deployment = Mock(return_value=deployment)

        callback = RateLimitCallback(router=router)
        result = callback._get_deployment_for_model("claude-3-sonnet")

        assert result == "deployment-123"

    def test_get_deployment_for_model_deployment_not_found(self):
        router = Mock()
        router.get_deployment = Mock(side_effect=Exception("Not found"))

        callback = RateLimitCallback(router=router)
        result = callback._get_deployment_for_model("claude-3-sonnet")

        assert result == "claude-3-sonnet"
