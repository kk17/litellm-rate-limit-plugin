"""Unit tests for RateLimitCallback."""

import asyncio
from unittest.mock import MagicMock, Mock

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
