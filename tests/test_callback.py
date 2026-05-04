"""Unit tests for RateLimitCallback."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_post_call_failure_500_triggers_cooldown(self):
        callback = RateLimitCallback()
        error = Mock()
        error.status_code = 500
        request_data = {"model": "claude-3-sonnet"}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_post_call_failure_402_triggers_cooldown(self):
        callback = RateLimitCallback()
        error = type("Error", (), {"status_code": 402, "headers": {}})()
        request_data = {"model": "github-copilot/claude-opus-4.6"}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("github-copilot/claude-opus-4.6")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_post_call_failure_401_skipped(self):
        callback = RateLimitCallback()
        error = type("Error", (), {"status_code": 401, "headers": {}})()
        request_data = {"model": "claude-3-sonnet"}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is False

    @pytest.mark.asyncio
    async def test_post_call_failure_no_status_code_skipped(self):
        callback = RateLimitCallback()
        error = Exception("connection reset")
        request_data = {"model": "claude-3-sonnet"}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is False

    @pytest.mark.asyncio
    async def test_update_cooldown_with_router(self):
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback = RateLimitCallback()
        callback.set_router(router)
        await callback._update_cooldown("claude-3-sonnet", 45.0, {})
        cooldown_cache.add_deployment_to_cooldown.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_cooldown_no_router(self):
        callback = RateLimitCallback()
        callback.set_router(None)
        await callback._update_cooldown("claude-3-sonnet", 45.0, {})

    @pytest.mark.asyncio
    async def test_update_cooldown_no_cache(self):
        router = Mock(spec=["cooldown_cache"])
        del router.cooldown_cache
        callback = RateLimitCallback()
        callback.set_router(router)
        await callback._update_cooldown("claude-3-sonnet", 45.0, {})

    def test_get_cooldown_for_model_default(self):
        callback = RateLimitCallback()
        assert callback._get_cooldown_for_model("unknown") == 60.0

    def test_get_deployment_ids_for_model_no_router(self):
        callback = RateLimitCallback()
        result = callback._get_deployment_ids_for_model("claude-3-sonnet")
        assert result == []

    def test_get_deployment_ids_for_model_with_dict_deployments(self):
        router = Mock()
        router.model_list = [
            {"model_name": "claude-3-sonnet", "model_info": {"id": "dep-123"}},
            {"model_name": "claude-3-opus", "model_info": {"id": "dep-456"}},
            {"model_name": "claude-3-sonnet", "model_info": {"id": "dep-789"}},
        ]
        callback = RateLimitCallback()
        callback.set_router(router)
        result = callback._get_deployment_ids_for_model("claude-3-sonnet")
        assert "dep-123" in result
        assert "dep-789" in result
        assert len(result) == 2

    def test_get_deployment_ids_for_model_no_match(self):
        router = Mock()
        router.model_list = [
            {"model_name": "claude-3-opus", "model_info": {"id": "dep-123"}},
        ]
        callback = RateLimitCallback()
        callback.set_router(router)
        result = callback._get_deployment_ids_for_model("claude-3-sonnet")
        assert result == []

    def test_get_model_names_from_router(self):
        router = Mock()
        router.model_list = [
            {"model_name": "claude-3-sonnet", "litellm_params": {"model": "anthropic/claude-3-sonnet"}},
            {"model_name": "claude-3-opus", "litellm_params": {"model": "anthropic/claude-3-opus"}},
            {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
        ]
        callback = RateLimitCallback()
        callback.set_router(router)
        result = callback._get_model_names_from_router()
        assert result == ["claude-3-opus", "claude-3-sonnet", "gpt-4"]

    def test_get_model_names_from_router_no_model_list(self):
        callback = RateLimitCallback()
        result = callback._get_model_names_from_router()
        assert result == []

    @pytest.mark.asyncio
    async def test_pre_call_hook_passes_healthy_model(self):
        callback = RateLimitCallback()
        data = {"model": "claude-3-sonnet"}
        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_pre_call_hook_does_not_raise_for_rate_limited(self):
        callback = RateLimitCallback()
        await callback._health_state.mark_rate_limited("claude-3-sonnet", 60.0)
        data = {"model": "claude-3-sonnet"}
        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_set_router_starts_health_checks(self):
        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        benchmark = HealthBenchmark(test_prompt="Say 'ok'")
        runner = HealthCheckRunner(benchmark=benchmark)

        router = Mock()
        router.model_list = [{"model_name": "claude-3-sonnet"}, {"model_name": "gpt-4"}]

        callback = RateLimitCallback(
            health_check_enabled=True,
            health_check_interval_seconds=60,
        )
        callback._health_runner = runner

        callback.set_router(router)

        await asyncio.sleep(0.1)

        assert runner.is_running("startup"), (
            "Health check task 'startup' should be running after set_router()"
        )

    @pytest.mark.asyncio
    async def test_ensure_router_starts_health_checks_lazily(self):
        from unittest.mock import patch

        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        benchmark = HealthBenchmark(test_prompt="Say 'ok'")
        runner = HealthCheckRunner(benchmark=benchmark)

        mock_router = Mock()
        mock_router.model_list = [{"model_name": "claude-3-sonnet"}]

        callback = RateLimitCallback(
            health_check_enabled=True,
            health_check_interval_seconds=60,
        )
        callback._health_runner = runner
        assert callback._health_checks_started is False

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            router = callback._ensure_router()

        assert router is mock_router
        assert callback._health_checks_started is True

        await asyncio.sleep(0.1)

        assert runner.is_running("startup")

    @pytest.mark.asyncio
    async def test_pre_call_hook_starts_health_checks_on_first_request(self):
        from unittest.mock import patch

        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        benchmark = HealthBenchmark(test_prompt="Say 'ok'")
        runner = HealthCheckRunner(benchmark=benchmark)

        mock_router = Mock()
        mock_router.model_list = [{"model_name": "gpt-4"}]

        callback = RateLimitCallback(
            health_check_enabled=True,
            health_check_interval_seconds=60,
        )
        callback._health_runner = runner

        assert callback._health_checks_started is False
        assert callback._router is None

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            data = {"model": "gpt-4"}
            result = await callback.async_pre_call_hook(
                user_api_key_dict=Mock(),
                cache=Mock(),
                data=data,
                call_type="completion",
            )

        assert result == data
        assert callback._health_checks_started is True
        assert callback._router is mock_router

        await asyncio.sleep(0.2)

        assert runner.is_running("startup"), (
            "Health checks should start when _ensure_router() obtains the router"
        )

    @pytest.mark.asyncio
    async def test_log_failure_event_triggers_cooldown(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        error = Mock()
        error.status_code = 402
        kwargs = {"model": "claude-3-sonnet", "exception": error}
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_log_failure_event_no_exception_skips(self):
        callback = RateLimitCallback()
        kwargs = {"model": "claude-3-sonnet"}
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is False

    @pytest.mark.asyncio
    async def test_log_failure_event_401_skipped(self):
        callback = RateLimitCallback()
        error = Mock()
        error.status_code = 401
        kwargs = {"model": "claude-3-sonnet", "exception": error}
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is False

    @pytest.mark.asyncio
    async def test_log_failure_event_rate_limit_with_headers(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        error = type("Error", (), {"status_code": 429, "headers": {"retry-after": "30"}})()
        kwargs = {"model": "claude-3-sonnet", "exception": error}
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("claude-3-sonnet")
        assert is_limited is True

    def test_build_model_mappings(self):
        router = Mock()
        router.model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/MiniMax-M2"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
        ]
        callback = RateLimitCallback()
        callback.set_router(router)

        assert callback._model_name_to_litellm_model == {
            "minimax-m2": "minimax/MiniMax-M2",
            "glm-5": "zai/glm-5",
        }

    def test_build_model_mappings_no_litellm_params(self):
        router = Mock()
        router.model_list = [
            {"model_name": "gpt-4"},
        ]
        callback = RateLimitCallback()
        callback.set_router(router)

        assert callback._model_name_to_litellm_model == {}

    @pytest.mark.asyncio
    async def test_health_check_client_uses_litellm_model(self):
        async def mock_acompletion(**kwargs):
            return None

        router = Mock()
        router.acompletion = mock_acompletion

        callback = RateLimitCallback()
        callback._router = router
        callback._model_name_to_litellm_model = {
            "minimax-m2": "minimax/MiniMax-M2",
        }

        client = callback._get_health_check_client()
        await client("minimax-m2", "test prompt")

    @pytest.mark.asyncio
    async def test_health_check_client_fallback_to_model_id(self):
        async def mock_acompletion(**kwargs):
            return None

        router = Mock()
        router.acompletion = mock_acompletion

        callback = RateLimitCallback()
        callback._router = router
        callback._model_name_to_litellm_model = {}

        client = callback._get_health_check_client()
        await client("unknown-model", "test prompt")

    @pytest.mark.asyncio
    async def test_log_failure_event_with_none_litellm_params(self):
        """Regression test: non-ProxyException errors with litellm_params=None must not crash."""
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = Mock()
        error.status_code = 500
        kwargs = {
            "model": "gpt-5.1",
            "exception": error,
            "litellm_params": None,
        }
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5.1")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_proxy_exception_skipped_in_log_failure_event(self):
        """ProxyException (auth/config errors) must NOT trigger cooldown."""
        from litellm.proxy._types import ProxyException

        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = ProxyException(
            message="No connected db.",
            type="no_db_connection",
            param=None,
            code=400,
        )
        kwargs = {
            "model": "gpt-5.1",
            "exception": error,
            "litellm_params": None,
        }
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5.1")
        assert is_limited is False
        cooldown_cache.add_deployment_to_cooldown.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_exception_skipped_in_post_call_failure_hook(self):
        """ProxyException must NOT trigger cooldown via post_call_failure_hook either."""
        from litellm.proxy._types import ProxyException

        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = ProxyException(
            message="Invalid API key",
            type="auth_error",
            param=None,
            code=401,
        )
        request_data = {"model": "gpt-5.1", "litellm_params": None}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5.1")
        assert is_limited is False
        cooldown_cache.add_deployment_to_cooldown.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_exception_invalid_api_key_skipped(self):
        """ProxyException for invalid API key (status 400) must NOT trigger cooldown."""
        from litellm.proxy._types import ProxyException

        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = ProxyException(
            message="invalid x-api-key",
            type="auth_error",
            param=None,
            code=400,
        )
        kwargs = {
            "model": "gpt-5-mini",
            "exception": error,
            "litellm_params": {
                "model_info": {"id": "38f9706c"},
                "model": "github_copilot/gpt-5-mini",
            },
        }
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5-mini")
        assert is_limited is False
        cooldown_cache.add_deployment_to_cooldown.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_failure_event_with_none_model_info(self):
        """Regression test: kwargs with model_info=None should not crash."""
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = Mock()
        error.status_code = 500
        kwargs = {
            "model": "gpt-5.1",
            "exception": error,
            "litellm_params": {"model_info": None, "model": "github_copilot/gpt-5.1"},
        }
        # Should NOT raise AttributeError
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5.1")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_post_call_failure_with_none_litellm_params_and_router(self):
        """Non-proxy 400 errors with litellm_params=None must not crash and must cooldown."""
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = Mock()
        error.status_code = 400
        request_data = {"model": "gpt-5.1", "litellm_params": None}
        await callback.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=error,
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-5.1")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_log_failure_event_empty_litellm_params(self):
        """Ensure empty dict litellm_params doesn't crash either."""
        callback = RateLimitCallback(default_cooldown_seconds=60.0)
        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock()
        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = []
        callback.set_router(router)

        error = Mock()
        error.status_code = 500
        kwargs = {
            "model": "gpt-4",
            "exception": error,
            "litellm_params": {},
        }
        await callback.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-4")
        assert is_limited is True

    def test_get_cooldown_for_model_none_litellm_params(self):
        callback = RateLimitCallback()
        request_data = {"litellm_params": None}
        assert callback._get_cooldown_for_model("gpt-5.1", request_data) == 60.0

    def test_get_cooldown_for_model_non_dict_litellm_params(self):
        callback = RateLimitCallback()
        request_data = {"litellm_params": "not_a_dict"}
        assert callback._get_cooldown_for_model("gpt-5.1", request_data) == 60.0

    @pytest.mark.asyncio
    async def test_sync_health_state_uses_remaining_ttl_not_default(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        cooldown_cache = MagicMock()
        add_mock = Mock()
        cooldown_cache.add_deployment_to_cooldown = add_mock
        cooldown_cache.get_active_cooldowns = Mock(return_value=[])

        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-123"}},
        ]
        callback.set_router(router)

        await callback._health_state.mark_rate_limited("glm-5.1", 5.0)

        await callback._sync_health_state_to_cooldown("glm-5.1")

        add_mock.assert_called_once()
        call_kwargs = add_mock.call_args[1]
        assert call_kwargs["cooldown_time"] < 6.0, (
            f"Expected cooldown_time close to 5.0s (remaining), got {call_kwargs['cooldown_time']}"
        )
        assert call_kwargs["cooldown_time"] >= 4.0, (
            f"Expected cooldown_time >= 4.0s (remaining minus drift), got {call_kwargs['cooldown_time']}"
        )

    @pytest.mark.asyncio
    async def test_sync_health_state_falls_back_to_default_when_no_health_entry(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        cooldown_cache = MagicMock()
        add_mock = Mock()
        cooldown_cache.add_deployment_to_cooldown = add_mock
        cooldown_cache.get_active_cooldowns = Mock(return_value=[])

        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-123"}},
        ]
        callback.set_router(router)

        await callback._sync_health_state_to_cooldown("glm-5.1")

        add_mock.assert_called_once()
        call_kwargs = add_mock.call_args[1]
        assert call_kwargs["cooldown_time"] == 60.0

    @pytest.mark.asyncio
    async def test_sync_uses_alias_state_remaining_when_health_state_has_none(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        cooldown_cache = MagicMock()
        add_mock = Mock()
        cooldown_cache.add_deployment_to_cooldown = add_mock
        cooldown_cache.get_active_cooldowns = Mock(return_value=[])

        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-123"}},
        ]
        callback.set_router(router)

        await callback._alias_state.mark_rate_limited("glm-5.1", 10.0)

        await callback._sync_health_state_to_cooldown("glm-5.1")

        add_mock.assert_called_once()
        call_kwargs = add_mock.call_args[1]
        assert call_kwargs["cooldown_time"] < 11.0
        assert call_kwargs["cooldown_time"] >= 9.0

    @pytest.mark.asyncio
    async def test_cooldown_auto_resumes_after_expiry(self):
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        _cooldown_entries: dict[str, float] = {}

        def mock_add(model_id, original_exception, exception_status, cooldown_time):
            _cooldown_entries[model_id] = time.monotonic() + cooldown_time

        def mock_get_active(model_ids, parent_otel_span=None):
            now = time.monotonic()
            return [mid for mid in model_ids if mid in _cooldown_entries and now < _cooldown_entries[mid]]

        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock(side_effect=mock_add)
        cooldown_cache.get_active_cooldowns = Mock(side_effect=mock_get_active)

        router = Mock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-123"}},
        ]
        callback.set_router(router)

        await callback._health_state.mark_rate_limited("glm-5.1", 1.0)

        data = {"model": "glm-5.1"}
        result = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data,
            call_type="completion",
        )
        assert result == data
        assert cooldown_cache.add_deployment_to_cooldown.call_count == 1

        await asyncio.sleep(1.1)

        is_limited = await callback._health_state.is_rate_limited("glm-5.1")
        assert is_limited is False

        _cooldown_entries.clear()
        data2 = {"model": "glm-5.1"}
        result2 = await callback.async_pre_call_hook(
            user_api_key_dict=Mock(),
            cache=Mock(),
            data=data2,
            call_type="completion",
        )
        assert result2 == data2
        assert cooldown_cache.add_deployment_to_cooldown.call_count == 1, (
            "Should NOT re-add to cooldown after rate limit expired"
        )


class TestPeriodicHealthCheckPersistence:
    """Tests for issue #0024: Health check should run periodically, not just once.

    The previous implementation closed the event loop immediately after initial
    checks completed, preventing periodic health checks from running.
    """

    @pytest.mark.asyncio
    async def test_run_initial_checks_and_start_periodic_creates_running_task(self):
        """Verify that initial checks start the periodic health check task."""
        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        runner = HealthCheckRunner(benchmark=HealthBenchmark())

        call_count = 0

        async def counting_client(model_id: str, prompt: str):
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": "ok"}}]}

        # Run initial checks and start periodic
        await runner.run_initial_checks_and_start_periodic(
            models=["model-a", "model-b"],
            interval_seconds=0.05,
            health_manager=None,
            client=counting_client,
            cooldown_seconds=60.0,
        )

        # After initial checks, the periodic task should be running
        assert runner.is_running("startup") is True, (
            "Periodic health check task should be running after initial checks complete"
        )

        # Wait for at least one more iteration with retry tolerance for slow CI
        for _ in range(10):
            await asyncio.sleep(0.05)
            if call_count >= 2:
                break

        # Should have at least 2 calls (initial + at least one periodic)
        assert call_count >= 2, (
            f"Expected at least 2 health check calls, got {call_count}. "
            "This indicates periodic checks are not running."
        )

        await runner.stop_all()

    @pytest.mark.asyncio
    async def test_health_check_task_runs_multiple_iterations(self):
        """Test that the health check task runs for multiple iterations."""
        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        runner = HealthCheckRunner(benchmark=HealthBenchmark())

        call_count = 0

        async def counting_client(model_id: str, prompt: str):
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": "ok"}}]}

        await runner.run_initial_checks_and_start_periodic(
            models=["model-x"],
            interval_seconds=0.1,
            health_manager=None,
            client=counting_client,
            cooldown_seconds=60.0,
        )

        # Wait for multiple iterations with retry tolerance for slow CI
        for _ in range(20):
            await asyncio.sleep(0.05)
            if call_count >= 3:
                break

        # Should have multiple calls
        assert call_count >= 3, (
            f"Expected at least 3 health check calls in 0.35s with 0.1s interval, got {call_count}. "
            "Periodic checks may not be running."
        )

        await runner.stop_all()

    @pytest.mark.asyncio
    async def test_health_runner_multiple_models_periodic(self):
        """Test periodic checks with multiple models."""
        from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner

        runner = HealthCheckRunner(benchmark=HealthBenchmark())

        calls = {"model-a": 0, "model-b": 0}

        async def tracking_client(model_id: str, prompt: str):
            calls[model_id] = calls.get(model_id, 0) + 1
            return {"choices": [{"message": {"content": "ok"}}]}

        await runner.run_initial_checks_and_start_periodic(
            models=["model-a", "model-b"],
            interval_seconds=0.1,
            health_manager=None,
            client=tracking_client,
            cooldown_seconds=60.0,
        )

        # Wait for multiple iterations with retry tolerance for slow CI
        for _ in range(20):
            await asyncio.sleep(0.05)
            if all(c >= 3 for c in calls.values()):
                break

        for model, count in calls.items():
            assert count >= 3, (
                f"Expected at least 3 health checks for {model}, got {count}. "
                "Periodic checks may not be running for all models."
            )

        await runner.stop_all()


class TestOriginalModelNameLogging:
    """Tests for issue #0025: Wired model name in invocation chain.

    When processing a request for alias model (e.g., `claude-sonnet-4-6`), the
    logs should show the original model name (`claude-sonnet-4-6`) rather than
    the resolved target model name (`a-glm-4.7`).
    """

    @pytest.mark.asyncio
    async def test_pre_call_hook_preserves_original_model_name_in_log(self):
        """Verify that pre-call hook uses original model name, not resolved alias."""
        from unittest.mock import MagicMock, patch

        router = MagicMock()
        router.model_list = []
        router.model_group_alias = {"claude-sonnet-4-6": "a-glm-4.7"}

        callback = RateLimitCallback()
        callback.set_router(router)

        data = {"model": "claude-sonnet-4-6"}

        with patch.object(
            callback._alias_state, "is_rate_limited", new_callable=AsyncMock
        ) as mock_is_limited:
            mock_is_limited.return_value = True
            result = await callback.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type="completion",
            )

        assert result == data
        # The original model name should be preserved
        assert mock_is_limited.call_args[0][0] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_async_post_call_success_logs_requested_model_not_alias(self):
        """Verify success hook logs the requested model, not the resolved one."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]
        router.model_group_alias = {"claude-sonnet-4-6": "a-minimax-m2.5"}

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {"a-minimax-m2.5": "minimax/MiniMax-M2.5"}

        response = MagicMock()
        response.model = "minimax/MiniMax-M2.5"

        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {},
        }

        with patch("litellm_rate_limit.callback.logger") as mock_logger:
            await callback.async_post_call_success_hook(
                data=data,
                response=response,
                user_api_key_dict=MagicMock(),
            )

            # The log should show the actual model used (a-minimax-m2.5) since it's different from requested
            info_calls = list(mock_logger.info.call_args_list)
            success_call = next(
                (c for c in info_calls if "Successfully called model" in str(c)),
                None,
            )
            assert success_call is not None, "Should have logged success message"
            # The actual model should be logged, not the alias
            assert "a-minimax-m2.5" in str(success_call)

    @pytest.mark.asyncio
    async def test_get_deployment_id_for_model_returns_correct_id(self):
        """Test that _get_deployment_id_for_model correctly returns deployment ID."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-glm-4.7", "model_info": {"id": "dep-glm-47"}},
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]

        callback = RateLimitCallback()
        callback.set_router(router)

        assert callback._get_deployment_id_for_model("a-glm-4.7") == "dep-glm-47"
        assert callback._get_deployment_id_for_model("a-minimax-m2.5") == "dep-minimax"
        assert callback._get_deployment_id_for_model("unknown") is None

    @pytest.mark.asyncio
    async def test_resolve_actual_model_with_deployment_returns_tuple(self):
        """Test that _resolve_actual_model_with_deployment returns (model, deployment_id)."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-glm-4.7", "model_info": {"id": "dep-glm-47"}},
        ]

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {"a-glm-4.7": "zai/glm-4.7"}

        response = MagicMock()

        # Test with litellm_model mapping
        data = {
            "model": "a-glm-4.7",
            "litellm_params": {"model": "zai/glm-4.7"},
        }

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="a-glm-4.7",
        )

        assert result == ("a-glm-4.7", "dep-glm-47")


class TestResolveActualModelFallbackPath:
    """Tests for issue #0026: Final call model is not correct in logging.

    When LiteLLM falls back to a different model, the success hook should
    log the actual model used, not the original requested model. The
    resolution uses the response.model field which contains the provider-prefixed
    model name of the actual model called.
    """

    @pytest.mark.asyncio
    async def test_resolve_uses_response_model_when_litellm_params_not_in_mapping(self):
        """Test that response.model is used when litellm_params.model is not in mapping."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]

        callback = RateLimitCallback()
        callback.set_router(router)
        # litellm_model mapping is empty - can't resolve from litellm_params
        callback._model_name_to_litellm_model = {}

        # Simulate a fallback response with provider-prefixed model
        response = MagicMock()
        response.model = "minimax/MiniMax-M2.5"

        data = {
            "model": "claude-sonnet-4-6",  # Original requested model (alias)
            "litellm_params": {"model": "claude-sonnet-4-6"},  # May not be in mapping
        }

        # Add the REVERSE mapping (litellm_model -> model_name) for response model resolution
        # This is needed because _resolve_model_from_litellm_model looks up by litellm_model value
        callback._model_name_to_litellm_model = {"a-minimax-m2.5": response.model}

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="claude-sonnet-4-6",
        )

        # Should resolve to a-minimax-m2.5 using response.model and get deployment ID
        assert result == ("a-minimax-m2.5", "dep-minimax")

    @pytest.mark.asyncio
    async def test_success_hook_logs_actual_fallback_model(self):
        """Test that async_post_call_success_hook logs actual model after fallback."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-glm-4.7", "model_info": {"id": "dep-glm-47"}},
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]
        router.model_group_alias = {"claude-sonnet-4-6": "a-minimax-m2.5"}

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {
            "a-minimax-m2.5": "minimax/MiniMax-M2.5",
        }

        # Response from fallback model
        response = MagicMock()
        response.model = "minimax/MiniMax-M2.5"

        data = {
            "model": "claude-sonnet-4-6",  # Original requested (alias)
            "litellm_params": {"model": "claude-sonnet-4-6"},
        }

        with patch("litellm_rate_limit.callback.logger") as mock_logger:
            await callback.async_post_call_success_hook(
                data=data,
                response=response,
                user_api_key_dict=MagicMock(),
            )

            # Should log the actual fallback model, not the original requested
            info_calls = list(mock_logger.info.call_args_list)
            success_call = next(
                (c for c in info_calls if "Successfully called model" in str(c)),
                None,
            )
            assert success_call is not None, "Should have logged success message"
            # The actual model (a-minimax-m2.5) should be logged
            assert "a-minimax-m2.5" in str(success_call)
            # The deployment ID should be shown, not "unknown"
            assert "dep-minimax" in str(success_call)

    @pytest.mark.asyncio
    async def test_response_model_equals_requested_but_deployment_id_resolves(self):
        """Regression test: response.model == requested_model should NOT return early.

        When a fallback happens but response.model equals the original requested
        model name (e.g. both are the alias "claude-sonnet-4-6"), the method should
        NOT return the alias as the actual model. Instead it should fall through to
        try litellm_params.model_info.id which can resolve the real model.
        """
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-glm-4.7", "model_info": {"id": "dep-glm-47"}},
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]
        router.model_group_alias = {"claude-sonnet-4-6": "a-glm-4.7"}

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {
            "a-glm-4.7": "zai/glm-4.7",
            "a-minimax-m2.5": "minimax/MiniMax-M2.5",
        }

        # response.model equals the alias (same as requested_model) — doesn't resolve
        response = MagicMock()
        response.model = "claude-sonnet-4-6"

        # litellm_params has model_info.id pointing to the fallback deployment
        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {
                "model_info": {"id": "dep-minimax"},
            },
        }

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="claude-sonnet-4-6",
        )

        # Should resolve via model_info.id, NOT return the alias
        assert result[0] == "a-minimax-m2.5", f"Expected actual model a-minimax-m2.5, got {result[0]}"
        assert result[1] == "dep-minimax", f"Expected deployment dep-minimax, got {result[1]}"

    @pytest.mark.asyncio
    async def test_response_model_as_model_name_gets_deployment_id(self):
        """Test that response.model is tried as a model_name to get deployment ID.

        When response.model contains a model_name (not a litellm_model), it should
        look up the deployment ID directly from the router's model_list.
        """
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]

        callback = RateLimitCallback()
        callback.set_router(router)
        # No litellm_model mapping — response.model is a model_name, not a litellm_model
        callback._model_name_to_litellm_model = {}

        response = MagicMock()
        response.model = "a-minimax-m2.5"

        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {},
        }

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="claude-sonnet-4-6",
        )

        assert result == ("a-minimax-m2.5", "dep-minimax")

    @pytest.mark.asyncio
    async def test_no_resolution_returns_requested_model_with_none(self):
        """When all resolution methods fail, return requested_model with None deployment."""
        router = MagicMock()
        router.model_list = []

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {}

        response = MagicMock()
        response.model = "unknown-model"

        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {},
        }

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="claude-sonnet-4-6",
        )

        # response.model != requested_model so it returns response.model, None
        assert result == ("unknown-model", None)

    @pytest.mark.asyncio
    async def test_no_response_model_uses_deployment_id_fallback(self):
        """When response has no model attribute, fall through to deployment ID lookup."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {}

        # Response without model attribute
        response = MagicMock(spec=[])

        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {
                "model_info": {"id": "dep-minimax"},
            },
        }

        result = callback._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model="claude-sonnet-4-6",
        )

        assert result == ("a-minimax-m2.5", "dep-minimax")

    @pytest.mark.asyncio
    async def test_success_hook_logs_via_deployment_id_when_response_model_is_alias(self):
        """End-to-end test: fallback success with response.model as alias resolves correctly."""
        router = MagicMock()
        router.model_list = [
            {"model_name": "a-glm-4.7", "model_info": {"id": "dep-glm-47"}},
            {"model_name": "a-minimax-m2.5", "model_info": {"id": "dep-minimax"}},
        ]
        router.model_group_alias = {"claude-sonnet-4-6": "a-glm-4.7"}

        callback = RateLimitCallback()
        callback.set_router(router)
        callback._model_name_to_litellm_model = {
            "a-glm-4.7": "zai/glm-4.7",
            "a-minimax-m2.5": "minimax/MiniMax-M2.5",
        }

        # Simulate: response.model is the alias (doesn't resolve), but deployment ID is available
        response = MagicMock()
        response.model = "claude-sonnet-4-6"

        data = {
            "model": "claude-sonnet-4-6",
            "litellm_params": {
                "model_info": {"id": "dep-minimax"},
            },
        }

        with patch("litellm_rate_limit.callback.logger") as mock_logger:
            await callback.async_post_call_success_hook(
                data=data,
                response=response,
                user_api_key_dict=MagicMock(),
            )

            info_calls = list(mock_logger.info.call_args_list)
            success_call = next(
                (c for c in info_calls if "Successfully called model" in str(c)),
                None,
            )
            assert success_call is not None, "Should have logged success message"
            # Should log the actual fallback model, not the alias
            assert "a-minimax-m2.5" in str(success_call), (
                f"Expected 'a-minimax-m2.5' in log, got: {success_call}"
            )
            assert "dep-minimax" in str(success_call), f"Expected 'dep-minimax' in log, got: {success_call}"


class TestCooldownCacheAutoRestore:
    """Tests for issue #0027: Rate limited model cannot be removed from cooldown.

    When a model's rate limit reset time passes, it should be automatically
    restored to health. Both the health state AND the cooldown_cache entry
    should be cleared so the model can receive requests again.
    """

    @pytest.mark.asyncio
    async def test_cooldown_cache_entry_expires_and_model_can_be_readded(self):
        """Test that after cooldown entry expires, model is healthy and not re-added."""
        from unittest.mock import MagicMock, Mock

        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        # Track cooldown entries with expiration
        _cooldown_entries: dict[str, float] = {}

        def mock_add(model_id, original_exception, exception_status, cooldown_time):
            _cooldown_entries[model_id] = time.monotonic() + cooldown_time

        def mock_get_active(model_ids, parent_otel_span=None):
            now = time.monotonic()
            return [mid for mid in model_ids if mid in _cooldown_entries and now < _cooldown_entries[mid]]

        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock(side_effect=mock_add)
        cooldown_cache.get_active_cooldowns = Mock(side_effect=mock_get_active)

        router = MagicMock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-glm-5.1"}},
        ]
        # Configure model_group_alias to return the model name itself for this model
        router.model_group_alias.get.return_value = "glm-5.1"
        callback.set_router(router)

        # Mark rate limited with short duration
        await callback._health_state.mark_rate_limited("glm-5.1", 1.0)

        # Pre-call should add to cooldown
        data = {"model": "glm-5.1"}
        await callback.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type="completion",
        )
        assert cooldown_cache.add_deployment_to_cooldown.call_count == 1
        assert "dep-glm-5.1" in _cooldown_entries

        # Wait for rate limit to expire
        await asyncio.sleep(1.1)

        # Health state should auto-restore
        is_limited = await callback._health_state.is_rate_limited("glm-5.1")
        assert is_limited is False, "Health state should auto-restore after rate limit expires"

        # Clear the cooldown_entries to simulate TTL expiration
        _cooldown_entries.clear()

        # Pre-call should NOT add to cooldown (model is healthy, not rate-limited)
        await callback.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data={"model": "glm-5.1"},
            call_type="completion",
        )
        # After health state entry expires, model is healthy, NOT re-added to cooldown
        assert cooldown_cache.add_deployment_to_cooldown.call_count == 0, (
            "Model should remain healthy after rate limit expires - should NOT be re-added to cooldown"
        )

    @pytest.mark.asyncio
    async def test_sync_health_state_to_cooldown_respects_expiry(self):
        """Test that _sync_health_state_to_cooldown doesn't re-add expired models."""
        from unittest.mock import MagicMock, Mock

        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        _cooldown_entries: dict[str, float] = {}

        def mock_add(model_id, original_exception, exception_status, cooldown_time):
            _cooldown_entries[model_id] = time.monotonic() + cooldown_time

        def mock_get_active(model_ids, parent_otel_span=None):
            now = time.monotonic()
            return [mid for mid in model_ids if mid in _cooldown_entries and now < _cooldown_entries[mid]]

        cooldown_cache = MagicMock()
        cooldown_cache.add_deployment_to_cooldown = Mock(side_effect=mock_add)
        cooldown_cache.get_active_cooldowns = Mock(side_effect=mock_get_active)

        router = MagicMock()
        router.cooldown_cache = cooldown_cache
        router.model_list = [
            {"model_name": "glm-5.1", "model_info": {"id": "dep-glm-5.1"}},
        ]
        callback.set_router(router)

        # Mark rate limited with short duration
        await callback._health_state.mark_rate_limited("glm-5.1", 1.0)

        # Initial sync to cooldown
        await callback._sync_health_state_to_cooldown("glm-5.1")
        assert "dep-glm-5.1" in _cooldown_entries

        # Wait for expiry
        await asyncio.sleep(1.1)

        # Health state should auto-restore
        is_limited = await callback._health_state.is_rate_limited("glm-5.1")
        assert is_limited is False

        # Clear the cooldown cache to simulate TTL expiration
        _cooldown_entries.clear()

        # Try to sync again - should NOT re-add since health state is restored
        await callback._sync_health_state_to_cooldown("glm-5.1")
        assert "dep-glm-5.1" not in _cooldown_entries

    @pytest.mark.asyncio
    async def test_alias_state_auto_restores_after_rate_limit(self):
        """Test that alias_state auto-restores model after rate limit expires."""
        callback = RateLimitCallback(default_cooldown_seconds=60.0)

        # Mark via alias_state (not health_state)
        await callback._alias_state.mark_rate_limited("glm-5.1", 1.0)

        # Should be rate limited
        is_limited = await callback._alias_state.is_rate_limited("glm-5.1")
        assert is_limited is True

        # Wait for expiry
        await asyncio.sleep(1.1)

        # Should auto-restore
        is_limited = await callback._alias_state.is_rate_limited("glm-5.1")
        assert is_limited is False
