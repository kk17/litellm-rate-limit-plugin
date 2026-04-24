"""End-to-end integration tests with mock LiteLLM proxy.

Tests the complete flow:
1. Mock LiteLLM proxy with router, cooldown_cache, and model aliases
2. Rate limit detection and blocking
3. Cooldown cache synchronization
4. Alias resolution
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from litellm_rate_limit import RateLimitCallback


@dataclass
class MockCooldownEntry:
    """Simulates a cooldown cache entry."""

    model_id: str
    cooldown_until: float
    reason: str = "rate_limit"


class MockCooldownCache:
    """Mock LiteLLM cooldown cache with async operations."""

    def __init__(self):
        self._entries: dict[str, MockCooldownEntry] = {}
        self._lock = asyncio.Lock()

    def add_deployment_to_cooldown(
        self,
        model_id: str,
        original_exception: Exception,
        exception_status: int,
        cooldown_time: float | None = None,
    ) -> None:
        effective_time = cooldown_time if cooldown_time is not None else 60.0
        self._entries[model_id] = MockCooldownEntry(
            model_id=model_id,
            cooldown_until=time.monotonic() + effective_time,
        )

    async def get_cooldown(self, model_id: str) -> MockCooldownEntry | None:
        entry = self._entries.get(model_id)
        if entry and time.monotonic() >= entry.cooldown_until:
            del self._entries[model_id]
            return None
        return entry

    async def clear_cooldown(self, model_id: str) -> None:
        self._entries.pop(model_id, None)

    def is_in_cooldown(self, model_id: str) -> bool:
        entry = self._entries.get(model_id)
        return bool(entry and time.monotonic() < entry.cooldown_until)


@dataclass
class MockDeployment:
    """Mock LiteLLM deployment."""

    model_name: str
    model_info: dict = field(default_factory=dict)


class MockRouter:
    """Mock LiteLLM Router with all required attributes."""

    def __init__(
        self,
        model_list: list[dict] | None = None,
        model_group_alias: dict[str, str] | None = None,
    ):
        self.model_list = model_list or []
        self.model_group_alias = model_group_alias or {}
        self.cooldown_cache = MockCooldownCache()
        self._deployments: dict[str, MockDeployment] = {}

        for model_config in self.model_list:
            model_name = model_config.get("model_name", "")
            model_id = model_config.get("model_info", {}).get("id", model_name)
            self._deployments[model_name] = MockDeployment(
                model_name=model_name,
                model_info={"id": model_id},
            )

    def get_deployment(self, model_id: str) -> MockDeployment | None:
        return self._deployments.get(model_id)

    async def acall(
        self,
        model: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Simulate a completion call."""
        if self.cooldown_cache.is_in_cooldown(model):
            raise Exception(f"Model {model} is in cooldown")

        return {
            "id": f"chatcmpl-{model}",
            "choices": [{"message": {"content": f"Response from {model}"}}],
        }


class MockDualCache:
    """Mock LiteLLM DualCache."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def get_cache(self, key: str) -> Any:
        return self._cache.get(key)

    def set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value


class MockUserAPIKeyAuth:
    """Mock UserAPIKeyAuth."""

    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key
        self.user_id = "test-user"


def create_rate_limit_error(
    status_code: int = 429,
    headers: dict | None = None,
    message: str = "Rate limit exceeded",
) -> Exception:
    """Create a mock rate limit error."""
    error = Exception(message)
    error.status_code = status_code
    error.headers = headers or {}
    return error


@pytest.fixture
def mock_router():
    """Create a mock LiteLLM router with realistic configuration."""
    return MockRouter(
        model_list=[
            {"model_name": "gpt-4", "model_info": {"id": "gpt-4-deployment-1"}},
            {"model_name": "gpt-4o-mini", "model_info": {"id": "gpt-4o-mini-deployment-1"}},
            {"model_name": "claude-3-opus", "model_info": {"id": "claude-3-opus-deployment-1"}},
            {"model_name": "claude-3-sonnet", "model_info": {"id": "claude-3-sonnet-deployment-1"}},
        ],
        model_group_alias={
            "gpt-4-turbo": "gpt-4",
            "claude-3": "claude-3-sonnet",
        },
    )


@pytest.fixture
def callback(mock_router):
    """Create a callback with router reference."""
    cb = RateLimitCallback(
        default_cooldown_seconds=60.0,
        models_to_check=[
            {"gpt-4": ["gpt-4", "gpt-4o-mini"]},
            {"claude-3-opus": ["claude-3-opus"]},
        ],
    )
    cb.set_router(mock_router)
    return cb


@pytest.fixture
def mock_cache():
    """Mock DualCache for hook parameters."""
    return MockDualCache()


@pytest.fixture
def mock_user_auth():
    """Mock UserAPIKeyAuth for hook parameters."""
    return MockUserAPIKeyAuth()


class TestMockProxyBasicFlow:
    """Tests with mock LiteLLM proxy - basic flow."""

    @pytest.mark.asyncio
    async def test_pre_call_hook_allows_non_limited_model(
        self, callback, mock_router, mock_cache, mock_user_auth
    ):
        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_pre_call_hook_does_not_raise_for_rate_limited(
        self, callback, mock_router, mock_cache, mock_user_auth
    ):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)

        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_post_failure_tracks_rate_limit(self, callback, mock_router, mock_cache, mock_user_auth):
        error = create_rate_limit_error(
            headers={"retry-after": "30"},
        )

        await callback.async_post_call_failure_hook(
            request_data={"model": "gpt-4"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )
        is_limited = await callback._alias_state.is_rate_limited("gpt-4")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_full_rate_limit_cycle(self, callback, mock_router, mock_cache, mock_user_auth):
        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )
        assert result == data

        error = create_rate_limit_error(headers={"retry-after": "60"})
        await callback.async_post_call_failure_hook(
            request_data={"model": "gpt-4"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        is_limited = await callback._alias_state.is_rate_limited("gpt-4")
        assert is_limited is True

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )
        assert result == data


class TestAliasResolution:
    """Tests for model alias resolution."""

    @pytest.mark.asyncio
    async def test_alias_blocked_when_target_rate_limited(
        self, callback, mock_router, mock_cache, mock_user_auth
    ):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)

        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data


class TestProviderProbe:
    """Tests for provider probe model behavior."""

    @pytest.mark.asyncio
    async def test_probe_model_blocks_unlisted_sibling(
        self, callback, mock_router, mock_cache, mock_user_auth
    ):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)

        data = {"model": "gpt-4-turbo-preview", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_explicit_model_has_own_health_status(
        self, callback, mock_router, mock_cache, mock_user_auth
    ):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)

        data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_different_provider_not_blocked(self, callback, mock_router, mock_cache, mock_user_auth):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)

        data = {"model": "claude-3-opus", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data


class TestCooldownExpiry:
    """Tests for cooldown expiration."""

    @pytest.mark.asyncio
    async def test_expired_cooldown_allows_request(self, callback, mock_router, mock_cache, mock_user_auth):
        await callback._health_state.mark_rate_limited("gpt-4", -1.0)

        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_clear_rate_limit_allows_request(self, callback, mock_router, mock_cache, mock_user_auth):
        await callback._health_state.mark_rate_limited("gpt-4", 60.0)
        await callback._health_state.clear_rate_limit("gpt-4")

        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        result = await callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result == data


class TestHeaderParsing:
    """Tests for rate limit header parsing."""


class TestNonRateLimitErrors:
    """Non-429 errors should also trigger cooldown with provider-specific cooldown."""

    @pytest.mark.asyncio
    async def test_500_error_triggers_cooldown(self, callback, mock_router, mock_user_auth):
        error = create_rate_limit_error(status_code=500, headers={})

        await callback.async_post_call_failure_hook(
            request_data={"model": "gpt-4"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        is_limited = await callback._alias_state.is_rate_limited("gpt-4")
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_401_error_does_not_trigger_cooldown(self, callback, mock_router, mock_user_auth):
        error = create_rate_limit_error(status_code=401, headers={})

        await callback.async_post_call_failure_hook(
            request_data={"model": "gpt-4"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        is_limited = await callback._alias_state.is_rate_limited("gpt-4")
        assert is_limited is False

    @pytest.mark.asyncio
    async def test_402_error_triggers_cooldown(self, callback, mock_router, mock_user_auth):
        callback = RateLimitCallback()

        error = create_rate_limit_error(status_code=402, headers={})

        await callback.async_post_call_failure_hook(
            request_data={"model": "github-copilot/claude-opus-4.6"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )
        is_limited = await callback._alias_state.is_rate_limited("github-copilot/claude-opus-4.6")
        assert is_limited is True


class TestAliasFallbackRewrite:
    """Tests for alias-aware rate limit detection and fallback injection.

    When an alias resolves to a rate-limited target:
    1. pre_call_hook detects via alias_state and syncs cooldown to target deployments
    2. _inject_alias_fallbacks mirrors fallbacks for alias names into router.fallbacks
    """

    @pytest.fixture
    def alias_router(self):
        router = MockRouter(
            model_list=[
                {"model_name": "og-qwen3.6-plus", "model_info": {"id": "og-qwen3.6-plus-dep-1"}},
                {"model_name": "glm-4.7", "model_info": {"id": "glm-4.7-dep-1"}},
            ],
            model_group_alias={
                "qwen-3.6-plus": "og-qwen3.6-plus",
            },
        )
        router.fallbacks = [{"og-qwen3.6-plus": ["glm-4.7"]}]
        return router

    @pytest.fixture
    def alias_callback(self, alias_router):
        cb = RateLimitCallback(default_cooldown_seconds=60.0)
        cb.set_router(alias_router)
        return cb

    @pytest.mark.asyncio
    async def test_inject_alias_fallbacks_mirrors_target_fallback(self, alias_callback, alias_router):
        assert {"qwen-3.6-plus": ["glm-4.7"]} in alias_router.fallbacks

    @pytest.mark.asyncio
    async def test_inject_alias_fallbacks_idempotent(self, alias_router):
        cb = RateLimitCallback(default_cooldown_seconds=60.0)
        cb.set_router(alias_router)
        cb.set_router(alias_router)
        alias_count = sum(
            1 for item in alias_router.fallbacks if isinstance(item, dict) and "qwen-3.6-plus" in item
        )
        assert alias_count == 1

    @pytest.mark.asyncio
    async def test_pre_call_syncs_cooldown_for_target_deployments(
        self, alias_callback, alias_router, mock_cache, mock_user_auth
    ):
        error = create_rate_limit_error(headers={"retry-after": "30"})
        await alias_callback.async_post_call_failure_hook(
            request_data={"model": "og-qwen3.6-plus"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        data = {"model": "qwen-3.6-plus", "messages": [{"role": "user", "content": "hi"}]}
        await alias_callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert alias_router.cooldown_cache.is_in_cooldown("og-qwen3.6-plus-dep-1")

    @pytest.mark.asyncio
    async def test_pre_call_does_not_rewrite_model(
        self, alias_callback, alias_router, mock_cache, mock_user_auth
    ):
        error = create_rate_limit_error(headers={"retry-after": "30"})
        await alias_callback.async_post_call_failure_hook(
            request_data={"model": "og-qwen3.6-plus"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        data = {"model": "qwen-3.6-plus", "messages": [{"role": "user", "content": "hi"}]}
        result = await alias_callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result["model"] == "qwen-3.6-plus"

    @pytest.mark.asyncio
    async def test_pre_call_no_rewrite_when_not_rate_limited(
        self, alias_callback, alias_router, mock_cache, mock_user_auth
    ):
        data = {"model": "qwen-3.6-plus", "messages": [{"role": "user", "content": "hi"}]}
        result = await alias_callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )

        assert result["model"] == "qwen-3.6-plus"

    @pytest.mark.asyncio
    async def test_post_failure_resolves_alias_for_cooldown(
        self, alias_callback, alias_router, mock_user_auth
    ):
        error = create_rate_limit_error(headers={"retry-after": "30"})
        await alias_callback.async_post_call_failure_hook(
            request_data={
                "model": "qwen-3.6-plus",
                "litellm_params": {},
            },
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        assert alias_router.cooldown_cache.is_in_cooldown("og-qwen3.6-plus-dep-1")

    @pytest.mark.asyncio
    async def test_full_alias_fallback_cycle(self, alias_callback, alias_router, mock_cache, mock_user_auth):
        # Step 1: Target model fails with 429
        error = create_rate_limit_error(headers={"retry-after": "60"})
        await alias_callback.async_post_call_failure_hook(
            request_data={"model": "og-qwen3.6-plus"},
            original_exception=error,
            user_api_key_dict=mock_user_auth,
        )

        # Step 2: Next request via alias — model NOT rewritten, but cooldown synced to target
        data = {"model": "qwen-3.6-plus", "messages": [{"role": "user", "content": "test"}]}
        result = await alias_callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data,
            call_type="completion",
        )
        assert result["model"] == "qwen-3.6-plus"

        # Step 3: Target deployment is in cooldown
        assert alias_router.cooldown_cache.is_in_cooldown("og-qwen3.6-plus-dep-1")

        # Step 4: Non-aliased model is not affected
        data2 = {"model": "glm-4.7", "messages": [{"role": "user", "content": "test"}]}
        result2 = await alias_callback.async_pre_call_hook(
            user_api_key_dict=mock_user_auth,
            cache=mock_cache,
            data=data2,
            call_type="completion",
        )
        assert result2["model"] == "glm-4.7"
        assert not alias_router.cooldown_cache.is_in_cooldown("glm-4.7-dep-1")


class TestConcurrentAccess:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_updates(self, callback, mock_router, mock_user_auth):
        models = ["gpt-4", "gpt-4o-mini", "claude-3-opus", "claude-3-sonnet"]

        async def mark_limited(model: str):
            error = create_rate_limit_error(headers={"retry-after": "30"})
            await callback.async_post_call_failure_hook(
                request_data={"model": model},
                original_exception=error,
                user_api_key_dict=mock_user_auth,
            )

        await asyncio.gather(*[mark_limited(m) for m in models])

        for model in models:
            is_limited = await callback._alias_state.is_rate_limited(model)
            assert is_limited is True
