"""Test for fallback behavior - verifies fallback models are available when primary is rate-limited."""

from unittest.mock import MagicMock, Mock

import pytest

from litellm_rate_limit.callback import RateLimitCallback
from litellm_rate_limit.parser import detect_api_error


class TestFallbackBehavior:
    @pytest.fixture
    def callback_with_router(self):
        callback = RateLimitCallback(default_cooldown_seconds=3600.0)
        router = Mock()
        cooldown_cache = MagicMock()
        cooldown_cache.get_active_cooldowns.return_value = []
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-glm-5.1"}},
            {"model_name": "og-glm-5.1", "model_info": {"id": "dep-og-glm-5.1"}},
            {"model_name": "a-k2p5", "model_info": {"id": "dep-a-k2p5"}},
        ]
        router.model_group_alias = {}
        callback.set_router(router)
        return callback

    @pytest.mark.asyncio
    async def test_primary_rate_limited_fallback_available(self, callback_with_router):
        callback = callback_with_router
        await callback._health_state.mark_rate_limited("glm-5.1", 3600.0)

        is_limited_primary = await callback._health_state.is_rate_limited("glm-5.1")
        is_limited_fallback = await callback._health_state.is_rate_limited("og-glm-5.1")

        assert is_limited_primary is True
        assert is_limited_fallback is False

    @pytest.mark.asyncio
    async def test_pre_call_cool_down_only_primary(self, callback_with_router):
        callback = callback_with_router
        await callback._health_state.mark_rate_limited("glm-5.1", 3600.0)

        data = {"model": "glm-5.1"}
        await callback.async_pre_call_hook(Mock(), Mock(), data, "completion")

        call_args_list = callback._router.cooldown_cache.add_deployment_to_cooldown.call_args_list
        deployment_ids = [call[1]["model_id"] for call in call_args_list]

        assert "dep-glm-5.1" in deployment_ids
        assert "dep-og-glm-5.1" not in deployment_ids

    @pytest.mark.asyncio
    async def test_alias_blocks_both(self):
        callback = RateLimitCallback(default_cooldown_seconds=3600.0)
        router = Mock()
        cooldown_cache = MagicMock()
        cooldown_cache.get_active_cooldowns.return_value = []
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-glm-5.1"}},
            {"model_name": "og-glm-5.1", "model_info": {"id": "dep-og-glm-5.1"}},
        ]
        router.model_group_alias = {"og-glm-5.1": "glm-5.1"}
        callback.set_router(router)

        await callback._alias_state.mark_rate_limited("glm-5.1", 3600.0)

        assert await callback._alias_state.is_rate_limited("glm-5.1") is True
        assert await callback._alias_state.is_rate_limited("og-glm-5.1") is True


class TestRouterRateLimitError:
    def test_detect_429_int(self):
        error = Mock()
        error.status_code = 429
        error.message = "Rate limit"

        result = detect_api_error(error)

        assert result is not None
        is_err, code, msg = result
        assert is_err is True
        assert code == 429

    def test_detect_429_string(self):
        error = Mock()
        error.status_code = "429"

        result = detect_api_error(error)

        assert result is not None
        is_err, code, msg = result
        assert is_err is True
