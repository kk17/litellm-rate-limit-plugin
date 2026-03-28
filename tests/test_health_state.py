"""Unit tests for HealthStateManager."""

import asyncio
from datetime import datetime, timezone

import pytest

from litellm_rate_limit.health_state import HealthStateManager, ModelHealthStatus


class TestModelHealthStatus:
    def test_model_health_status_creation(self):
        status = ModelHealthStatus(model_id="claude-3-sonnet")
        assert status.model_id == "claude-3-sonnet"
        assert status.is_rate_limited is False
        assert status.consecutive_failures == 0


class TestHealthStateManager:
    def test_init(self):
        manager = HealthStateManager()
        assert manager._rate_limited_until == {}
        assert manager._rate_limit_reset_at == {}
        assert manager._model_status == {}

    @pytest.mark.asyncio
    async def test_mark_rate_limited(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("claude-3-sonnet", 60.0)

        assert "claude-3-sonnet" in manager._rate_limited_until
        status = await manager.get_model_status("claude-3-sonnet")
        assert status is not None
        assert status.is_rate_limited is True

    @pytest.mark.asyncio
    async def test_mark_rate_limited_with_reset_time(self):
        manager = HealthStateManager()
        reset_at = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        await manager.mark_rate_limited("claude-3-sonnet", 60.0, reset_at=reset_at)

        assert manager._rate_limit_reset_at["claude-3-sonnet"] == reset_at

    @pytest.mark.asyncio
    async def test_is_rate_limited_true(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("claude-3-sonnet", 60.0)

        result = await manager.is_rate_limited("claude-3-sonnet")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_rate_limited_false(self):
        manager = HealthStateManager()
        result = await manager.is_rate_limited("claude-3-sonnet")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_rate_limited_expired(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("claude-3-sonnet", -1.0)

        result = await manager.is_rate_limited("claude-3-sonnet")
        assert result is False

        assert "claude-3-sonnet" not in manager._rate_limited_until

    @pytest.mark.asyncio
    async def test_get_healthy_models_filters(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("model-a", 60.0)

        all_models = ["model-a", "model-b", "model-c"]
        healthy = await manager.get_healthy_models(all_models)

        assert "model-a" not in healthy
        assert "model-b" in healthy
        assert "model-c" in healthy

    @pytest.mark.asyncio
    async def test_get_rate_limited_models(self):
        manager = HealthStateManager()
        reset_at = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        await manager.mark_rate_limited("model-a", 60.0, reset_at=reset_at)
        await manager.mark_rate_limited("model-b", 120.0)

        limited = await manager.get_rate_limited_models()

        assert "model-a" in limited
        assert limited["model-a"] == reset_at

    @pytest.mark.asyncio
    async def test_get_rate_limited_models_excludes_expired(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("expired-model", -1.0)
        await manager.mark_rate_limited("active-model", 60.0)

        limited = await manager.get_rate_limited_models()

        assert "expired-model" not in limited

    @pytest.mark.asyncio
    async def test_clear_rate_limit(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("claude-3-sonnet", 60.0)

        result = await manager.clear_rate_limit("claude-3-sonnet")
        assert result is True
        assert await manager.is_rate_limited("claude-3-sonnet") is False

    @pytest.mark.asyncio
    async def test_clear_rate_limit_not_found(self):
        manager = HealthStateManager()
        result = await manager.clear_rate_limit("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_model_status(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("claude-3-sonnet", 60.0)

        status = await manager.get_model_status("claude-3-sonnet")
        assert status is not None
        assert status.model_id == "claude-3-sonnet"
        assert status.is_rate_limited is True

    @pytest.mark.asyncio
    async def test_get_model_status_nonexistent(self):
        manager = HealthStateManager()
        status = await manager.get_model_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_record_failure(self):
        manager = HealthStateManager()
        await manager.record_failure("claude-3-sonnet", "Connection timeout")

        status = await manager.get_model_status("claude-3-sonnet")
        assert status is not None
        assert status.consecutive_failures == 1
        assert status.last_error == "Connection timeout"

    @pytest.mark.asyncio
    async def test_record_failure_multiple(self):
        manager = HealthStateManager()
        await manager.record_failure("claude-3-sonnet", "Error 1")
        await manager.record_failure("claude-3-sonnet", "Error 2")

        status = await manager.get_model_status("claude-3-sonnet")
        assert status is not None
        assert status.consecutive_failures == 2
        assert status.last_error == "Error 2"

    @pytest.mark.asyncio
    async def test_record_success(self):
        manager = HealthStateManager()
        await manager.record_failure("claude-3-sonnet", "Error")
        await manager.record_success("claude-3-sonnet")

        status = await manager.get_model_status("claude-3-sonnet")
        assert status is not None
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.last_check_time is not None

    @pytest.mark.asyncio
    async def test_clear_all(self):
        manager = HealthStateManager()
        await manager.mark_rate_limited("model-a", 60.0)
        await manager.mark_rate_limited("model-b", 120.0)

        await manager.clear_all()

        assert await manager.is_rate_limited("model-a") is False
        assert await manager.is_rate_limited("model-b") is False

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        manager = HealthStateManager()

        async def mark_model(model: str, times: int):
            for i in range(times):
                await manager.mark_rate_limited(f"{model}-{i}", 60.0)

        await asyncio.gather(
            mark_model("model-a", 10),
            mark_model("model-b", 10),
        )

        for i in range(10):
            assert await manager.is_rate_limited(f"model-a-{i}") is True
            assert await manager.is_rate_limited(f"model-b-{i}") is True
