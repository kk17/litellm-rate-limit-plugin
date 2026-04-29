from unittest.mock import MagicMock, Mock

import pytest

from litellm_rate_limit.callback import RateLimitCallback


def _make_callback():
    callback = RateLimitCallback(default_cooldown_seconds=3600.0)
    router = Mock()
    cooldown_cache = MagicMock()
    active_cooldowns: set[str] = set()

    def _mock_add_deployment(model_id, **kwargs):
        active_cooldowns.add(model_id)

    def _mock_get_active(model_ids, **kwargs):
        return [(mid, {}) for mid in model_ids if mid in active_cooldowns]

    cooldown_cache.add_deployment_to_cooldown.side_effect = _mock_add_deployment
    cooldown_cache.get_active_cooldowns.side_effect = _mock_get_active
    router.cooldown_cache = cooldown_cache
    router._active_cooldowns = active_cooldowns
    router.model_list = [
        {
            "model_name": "gpt-5.1",
            "litellm_params": {"model": "github_copilot/gpt-5.1"},
            "model_info": {"id": "dep-gpt-5.1"},
        },
        {
            "model_name": "gpt-5",
            "litellm_params": {"model": "github_copilot/gpt-5"},
            "model_info": {"id": "dep-gpt-5"},
        },
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {"model": "github_copilot/gpt-5-mini"},
            "model_info": {"id": "dep-gpt-5-mini"},
        },
    ]
    router.model_group_alias = {}
    router.fallbacks = [{"gpt-5.1": ["gpt-5", "gpt-5-mini"]}]
    callback.set_router(router)
    return callback


def _get_cooldown_deployment_ids(cb):
    return [
        call[1]["model_id"] for call in cb._router.cooldown_cache.add_deployment_to_cooldown.call_args_list
    ]


def _reset_cooldown_mock(cb):
    cb._router.cooldown_cache.reset_mock()
    active: set[str] = cb._router._active_cooldowns
    active.clear()

    def _mock_add_deployment(model_id, **kwargs):
        active.add(model_id)

    def _mock_get_active(model_ids, **kwargs):
        return [(mid, {}) for mid in model_ids if mid in active]

    cb._router.cooldown_cache.add_deployment_to_cooldown.side_effect = _mock_add_deployment
    cb._router.cooldown_cache.get_active_cooldowns.side_effect = _mock_get_active


class TestPreCallHookNoRaise:
    @pytest.mark.asyncio
    async def test_rate_limited_model_does_not_raise(self):
        cb = _make_callback()
        await cb._health_state.mark_rate_limited("gpt-5.1", 3600.0)

        data = {"model": "gpt-5.1"}
        result = await cb.async_pre_call_hook(Mock(), Mock(), data, "completion")
        assert result == data

    @pytest.mark.asyncio
    async def test_rate_limited_model_added_to_cooldown(self):
        cb = _make_callback()
        await cb._health_state.mark_rate_limited("gpt-5.1", 3600.0)

        await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")

        call_args = cb._router.cooldown_cache.add_deployment_to_cooldown.call_args
        assert call_args[1]["model_id"] == "dep-gpt-5.1"
        assert call_args[1]["cooldown_time"] == pytest.approx(3600.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_healthy_model_no_cooldown(self):
        cb = _make_callback()
        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5-mini"}, "completion")
        assert result == {"model": "gpt-5-mini"}
        assert not cb._router.cooldown_cache.add_deployment_to_cooldown.called

    @pytest.mark.asyncio
    async def test_multiple_calls_skip_duplicate_cooldown(self):
        cb = _make_callback()
        await cb._health_state.mark_rate_limited("gpt-5.1", 3600.0)

        for _ in range(3):
            result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
            assert result is not None

        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 1


class TestFailureHookDetectsNon429:
    @pytest.mark.asyncio
    async def test_plain_exception_with_error_code_400(self):
        cb = _make_callback()
        exc = Exception("Error code: 400 - {'error': {'message': 'The requested model is not supported.'}}")

        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5"},
            original_exception=exc,
            user_api_key_dict=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

    @pytest.mark.asyncio
    async def test_litellm_bad_request_error_detected(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()
        exc = BadRequestError(
            message="Github_copilotException - The requested model is not supported.",
            llm_provider="github_copilot",
            model="gpt-5",
        )

        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5"},
            original_exception=exc,
            user_api_key_dict=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

    @pytest.mark.asyncio
    async def test_401_error_skipped(self):
        from litellm.exceptions import AuthenticationError

        cb = _make_callback()
        exc = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-5",
        )

        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5"},
            original_exception=exc,
            user_api_key_dict=Mock(),
        )

        assert not await cb._alias_state.is_rate_limited("gpt-5")
        assert not cb._router.cooldown_cache.add_deployment_to_cooldown.called


class TestFullFallbackChain:
    @pytest.mark.asyncio
    async def test_second_call_both_models_in_cooldown(self):
        cb = _make_callback()

        await cb._health_state.mark_rate_limited("gpt-5.1", 3600.0)

        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
        assert result is not None
        assert "dep-gpt-5.1" in _get_cooldown_deployment_ids(cb)

        from litellm.exceptions import BadRequestError

        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
            user_api_key_dict=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

        _reset_cooldown_mock(cb)

        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
        assert result is not None

        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5.1" in dep_ids
        assert "dep-gpt-5-mini" not in dep_ids

    @pytest.mark.asyncio
    async def test_repeated_calls_no_exception(self):
        cb = _make_callback()
        await cb._health_state.mark_rate_limited("gpt-5.1", 3600.0)

        from litellm.exceptions import BadRequestError

        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
            user_api_key_dict=Mock(),
        )

        for _ in range(5):
            result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
            assert result is not None
            assert "model" in result


class TestLogFailureEventPath:
    @pytest.mark.asyncio
    async def test_log_failure_event_with_bad_request_error(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()
        kwargs = {
            "model": "gpt-5",
            "exception": BadRequestError(
                message="Github_copilotException - The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
        }

        await cb.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=Mock(),
            end_time=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

    @pytest.mark.asyncio
    async def test_log_failure_event_with_openai_error(self):
        from litellm.llms.openai.common_utils import OpenAIError

        cb = _make_callback()
        exc = OpenAIError(
            status_code=400,
            message="Error code: 400 - {'error': {'message': 'The requested model is not supported.'}}",
        )

        kwargs = {"model": "gpt-5", "exception": exc}

        await cb.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=Mock(),
            end_time=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

    @pytest.mark.asyncio
    async def test_log_failure_event_no_exception_skips(self):
        cb = _make_callback()
        kwargs = {"model": "gpt-5"}

        await cb.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=Mock(),
            end_time=Mock(),
        )

        assert not await cb._alias_state.is_rate_limited("gpt-5")
        assert not cb._router.cooldown_cache.add_deployment_to_cooldown.called


class TestResult4ExactScenario:
    @pytest.mark.asyncio
    async def test_second_call_no_wasted_api_call(self):
        cb = _make_callback()

        # --- First call: gpt-5.1 → 429, gpt-5 → BadRequestError, gpt-5-mini → success ---
        from litellm.exceptions import BadRequestError, RateLimitError

        # gpt-5.1 gets 429
        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5.1"},
            original_exception=RateLimitError(
                message="Rate limit exceeded",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            user_api_key_dict=Mock(),
        )
        assert await cb._alias_state.is_rate_limited("gpt-5.1")

        # gpt-5 gets BadRequestError — simulates async_log_failure_event path
        await cb.async_log_failure_event(
            kwargs={
                "model": "gpt-5",
                "exception": BadRequestError(
                    message="Github_copilotException - The requested model is not supported.",
                    llm_provider="github_copilot",
                    model="gpt-5",
                ),
            },
            response_obj=None,
            start_time=Mock(),
            end_time=Mock(),
        )
        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

        # --- Second call: gpt-5.1 → pre-call hook adds to cooldown, no exception raised ---
        _reset_cooldown_mock(cb)

        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
        assert result is not None

        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5.1" in dep_ids
        assert "dep-gpt-5-mini" not in dep_ids

        # gpt-5 is already in alias_state cooldown from the first call
        # LiteLLM router will see gpt-5 in cooldown cache and skip it
        # → goes directly to gpt-5-mini → success → no exception at all


class TestLogFailureFallbackEvent:
    """Tests for log_failure_fallback_event — the hook that captures fallback
    model failures during LiteLLM Router's fallback chain.

    This hook bypasses the has_logged_async_failure dedup that blocks
    async_log_failure_event for fallback model failures.
    """

    @pytest.mark.asyncio
    async def test_captures_fallback_model_failure(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        original_exc = BadRequestError(
            message="The requested model is not supported.",
            llm_provider="github_copilot",
            model="gpt-5.1",
        )
        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5"},
            original_exception=original_exc,
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        assert "dep-gpt-5" in _get_cooldown_deployment_ids(cb)

    @pytest.mark.asyncio
    async def test_skips_already_rate_limited_model(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()
        await cb._alias_state.mark_rate_limited("gpt-5", 3600.0)

        original_exc = BadRequestError(
            message="The requested model is not supported.",
            llm_provider="github_copilot",
            model="gpt-5.1",
        )
        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5"},
            original_exception=original_exc,
        )

        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_empty_model(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={},
            original_exception=BadRequestError(
                message="error",
                llm_provider="openai",
                model="gpt-5.1",
            ),
        )

        assert not await cb._alias_state.is_rate_limited("gpt-5.1")
        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 0


class TestLogSuccessFallbackEvent:
    """Tests for log_success_fallback_event — infers failed models from the
    fallback chain when a fallback succeeds."""

    @pytest.mark.asyncio
    async def test_infers_intermediate_failures(self):
        cb = _make_callback()

        await cb.log_success_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5-mini"},
            original_exception=Exception("original failure"),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5" in dep_ids
        assert "dep-gpt-5-mini" not in dep_ids

    @pytest.mark.asyncio
    async def test_skips_if_same_model(self):
        cb = _make_callback()

        await cb.log_success_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5.1"},
            original_exception=Exception("original failure"),
        )

        assert not await cb._alias_state.is_rate_limited("gpt-5.1")
        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_already_rate_limited(self):
        cb = _make_callback()
        await cb._alias_state.mark_rate_limited("gpt-5", 3600.0)

        await cb.log_success_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5-mini"},
            original_exception=Exception("original failure"),
        )

        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 0


class TestResult7ExactScenario:
    """Reproduce the exact result7.txt scenario.

    First call:
      1. gpt-5.1 → BadRequestError → async_post_call_failure_hook captures → cooldown
      2. Fallback to gpt-5 → BadRequestError → log_failure_fallback_event captures → cooldown
      3. Fallback to gpt-5-mini → success → log_success_fallback_event fires (safety net)

    Second call:
      1. Pre-call hook: gpt-5.1 rate-limited → synced to cooldown
      2. LiteLLM skips gpt-5.1 (cooldown) → tries gpt-5 → ALREADY in cooldown → skipped
      3. LiteLLM tries gpt-5-mini → success → no exception/traceback at all
    """

    @pytest.mark.asyncio
    async def test_first_call_captures_all_failures(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        # Step 1: gpt-5.1 BadRequestError — captured by async_post_call_failure_hook
        _reset_cooldown_mock(cb)
        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5.1"},
            original_exception=BadRequestError(
                message="Github_copilotException - The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            user_api_key_dict=Mock(),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5.1")
        assert not await cb._alias_state.is_rate_limited("gpt-5")

        # Step 2: gpt-5 BadRequestError during fallback — captured by log_failure_fallback_event
        _reset_cooldown_mock(cb)
        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5"},
            original_exception=BadRequestError(
                message="Github_copilotException - The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
        )

        assert await cb._alias_state.is_rate_limited("gpt-5")
        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5" in dep_ids

        # Step 3: gpt-5-mini succeeds — log_success_fallback_event fires but gpt-5 already cooldown
        _reset_cooldown_mock(cb)
        await cb.log_success_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5-mini"},
            original_exception=Exception("original"),
        )
        assert cb._router.cooldown_cache.add_deployment_to_cooldown.call_count == 0

    @pytest.mark.asyncio
    async def test_second_call_no_wasted_api_calls(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()
        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5.1"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            user_api_key_dict=Mock(),
        )

        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
        )

        # === Second call: pre-call hooks sync both to cooldown cache ===
        _reset_cooldown_mock(cb)

        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
        assert result is not None

        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5.1" in dep_ids

        assert await cb._alias_state.is_rate_limited("gpt-5")

        result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5"}, "completion")
        assert result is not None
        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5" in dep_ids

    @pytest.mark.asyncio
    async def test_repeated_calls_stable(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        # First call: both fail
        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5.1"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            user_api_key_dict=Mock(),
        )
        await cb.log_failure_fallback_event(
            original_model_group="gpt-5.1",
            kwargs={"model": "gpt-5"},
            original_exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5",
            ),
        )

        # Subsequent calls — no exceptions, both models in cooldown
        for _ in range(5):
            result = await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")
            assert result is not None
            assert "model" in result

        # gpt-5-mini should never be in cooldown
        assert not await cb._alias_state.is_rate_limited("gpt-5-mini")


class TestResult5ExactScenario:
    """Reproduce the exact result5.txt scenario.

    First call:
      1. gpt-5.1 → BadRequestError → detected → added to cooldown
      2. Fallback to gpt-5 → API call → BadRequestError → detected → added to cooldown
      3. Fallback to gpt-5-mini → success → HTTP 200

    Second call:
      1. Pre-call hook: gpt-5.1 rate-limited → synced to cooldown
      2. gpt-5 is NOT synced to cooldown in pre-call hook (only primary model)
      3. LiteLLM skips gpt-5.1 (cooldown) → tries gpt-5 → API call → BadRequestError
         → our failure hook captures it → adds to cooldown for NEXT time
      4. Fallback to gpt-5-mini → success

    This avoids the RouterRateLimitError issue where putting gpt-5 in cooldown
    preemptively causes LiteLLM to throw RouterRateLimitError which breaks the
    fallback chain and prevents gpt-5-mini from being tried.
    """

    @pytest.mark.asyncio
    async def test_first_call_captures_both_failures(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        # --- First call: gpt-5.1 → BadRequestError ---
        _reset_cooldown_mock(cb)
        await cb.async_post_call_failure_hook(
            request_data={"model": "gpt-5.1"},
            original_exception=BadRequestError(
                message="Github_copilotException - The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            user_api_key_dict=Mock(),
        )

        # Only gpt-5.1 should be in cooldown (no cascade)
        dep_ids = _get_cooldown_deployment_ids(cb)
        assert "dep-gpt-5.1" in dep_ids
        assert "dep-gpt-5" not in dep_ids, "gpt-5 should NOT be cascaded to cooldown"
        assert "dep-gpt-5-mini" not in dep_ids, "gpt-5-mini must NOT be in cooldown"

        # Verify gpt-5.1 is known unhealthy
        assert await cb._alias_state.is_rate_limited("gpt-5.1")
        assert not await cb._alias_state.is_rate_limited("gpt-5")
        assert not await cb._alias_state.is_rate_limited("gpt-5-mini")

    @pytest.mark.asyncio
    async def test_second_call_only_primary_model_in_cooldown(self):
        from litellm.exceptions import BadRequestError

        cb = _make_callback()

        # First call: gpt-5.1 fails
        await cb._handle_deployment_failure(
            exception=BadRequestError(
                message="The requested model is not supported.",
                llm_provider="github_copilot",
                model="gpt-5.1",
            ),
            model="gpt-5.1",
            request_data={"model": "gpt-5.1"},
        )

        # Verify only gpt-5.1 is marked unhealthy (no cascade)
        assert await cb._alias_state.is_rate_limited("gpt-5.1")
        assert not await cb._alias_state.is_rate_limited("gpt-5")

        # Second call: pre-call hook syncs only primary model
        _reset_cooldown_mock(cb)
        await cb.async_pre_call_hook(Mock(), Mock(), {"model": "gpt-5.1"}, "completion")

        dep_ids = _get_cooldown_deployment_ids(cb)
        # Only gpt-5.1 deployment is in cooldown
        # gpt-5 is NOT in cooldown → LiteLLM will try it (and our failure hook will catch it)
        assert "dep-gpt-5.1" in dep_ids
        assert "dep-gpt-5" not in dep_ids, "gpt-5 should NOT be in cooldown - let LiteLLM try it"
        assert "dep-gpt-5-mini" not in dep_ids
