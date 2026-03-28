"""Unit tests for AliasAwareHealthState."""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from litellm_rate_limit.alias_aware_state import AliasAwareHealthState, RateLimitEntry


class TestRateLimitEntry:
    def test_rate_limit_entry_creation(self):
        entry = RateLimitEntry(
            model_id="claude-3-sonnet",
            until_monotonic=time.monotonic() + 60,
            reset_at=datetime.now(timezone.utc),
        )
        assert entry.model_id == "claude-3-sonnet"
        assert entry.reset_at is not None


class TestAliasAwareHealthState:
    def test_init_default_values(self):
        state = AliasAwareHealthState()
        assert state.router is None
        assert state._rate_limited_targets == {}

    def test_set_router(self):
        state = AliasAwareHealthState()
        router = Mock()
        state.set_router(router)
        assert state.router == router

    def test_resolve_alias_to_target(self):
        router = Mock()
        router.model_group_alias = {"claude-opus-4-6": "claude-opus-4.6"}

        state = AliasAwareHealthState(router=router)
        result = state._resolve_to_target("claude-opus-4-6")

        assert result == "claude-opus-4.6"

    def test_resolve_target_returns_self(self):
        router = Mock()
        router.model_group_alias = {"claude-opus-4-6": "claude-opus-4.6"}

        state = AliasAwareHealthState(router=router)
        result = state._resolve_to_target("claude-opus-4.6")

        assert result == "claude-opus-4.6"

    def test_resolve_no_router_returns_self(self):
        state = AliasAwareHealthState()
        result = state._resolve_to_target("claude-3-sonnet")
        assert result == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_mark_rate_limited_uses_target(self):
        router = Mock()
        router.model_group_alias = {"alias-model": "target-model"}

        state = AliasAwareHealthState(router=router)
        await state.mark_rate_limited("alias-model", 60.0)

        assert "target-model" in state._rate_limited_targets
        assert "alias-model" not in state._rate_limited_targets

    @pytest.mark.asyncio
    async def test_is_rate_limited_checks_target(self):
        router = Mock()
        router.model_group_alias = {"alias-model": "target-model"}

        state = AliasAwareHealthState(router=router)
        await state.mark_rate_limited("alias-model", 60.0)

        assert await state.is_rate_limited("alias-model") is True
        assert await state.is_rate_limited("target-model") is True

    @pytest.mark.asyncio
    async def test_is_rate_limited_false(self):
        state = AliasAwareHealthState()
        result = await state.is_rate_limited("claude-3-sonnet")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_rate_limited_expired(self):
        state = AliasAwareHealthState()
        await state.mark_rate_limited("claude-3-sonnet", -1.0)

        result = await state.is_rate_limited("claude-3-sonnet")
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_aliases_same_target(self):
        router = Mock()
        router.model_group_alias = {
            "alias1": "target-model",
            "alias2": "target-model",
            "alias3": "other-model",
        }

        state = AliasAwareHealthState(router=router)
        await state.mark_rate_limited("alias1", 60.0)

        assert await state.is_rate_limited("alias1") is True
        assert await state.is_rate_limited("alias2") is True
        assert await state.is_rate_limited("alias3") is False

    @pytest.mark.asyncio
    async def test_get_healthy_models_filters(self):
        router = Mock()
        router.model_group_alias = {"blocked-alias": "blocked-target"}

        state = AliasAwareHealthState(router=router)
        await state.mark_rate_limited("blocked-alias", 60.0)

        all_models = ["healthy-model", "blocked-alias", "another-healthy"]
        healthy = await state.get_healthy_models(all_models)

        assert "healthy-model" in healthy
        assert "another-healthy" in healthy
        assert "blocked-alias" not in healthy

    @pytest.mark.asyncio
    async def test_clear_rate_limit(self):
        state = AliasAwareHealthState()
        await state.mark_rate_limited("claude-3-sonnet", 60.0)

        result = await state.clear_rate_limit("claude-3-sonnet")
        assert result is True
        assert await state.is_rate_limited("claude-3-sonnet") is False

    @pytest.mark.asyncio
    async def test_clear_rate_limit_not_found(self):
        state = AliasAwareHealthState()
        result = await state.clear_rate_limit("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_all_rate_limited(self):
        state = AliasAwareHealthState()
        await state.mark_rate_limited("model-a", 60.0)
        await state.mark_rate_limited("model-b", 120.0)

        all_limited = await state.get_all_rate_limited()

        assert "model-a" in all_limited
        assert "model-b" in all_limited
        assert len(all_limited) == 2

    @pytest.mark.asyncio
    async def test_get_all_rate_limited_excludes_expired(self):
        state = AliasAwareHealthState()
        await state.mark_rate_limited("expired-model", -1.0)
        await state.mark_rate_limited("active-model", 60.0)

        all_limited = await state.get_all_rate_limited()

        assert "expired-model" not in all_limited
        assert "active-model" in all_limited

    @pytest.mark.asyncio
    async def test_clear_all(self):
        state = AliasAwareHealthState()
        await state.mark_rate_limited("model-a", 60.0)
        await state.mark_rate_limited("model-b", 120.0)

        await state.clear_all()

        assert await state.is_rate_limited("model-a") is False
        assert await state.is_rate_limited("model-b") is False

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        state = AliasAwareHealthState()

        async def mark_model(model: str, times: int):
            for i in range(times):
                await state.mark_rate_limited(f"{model}-{i}", 60.0)

        await asyncio.gather(
            mark_model("model-a", 10),
            mark_model("model-b", 10),
        )

        all_limited = await state.get_all_rate_limited()
        assert len(all_limited) == 20
