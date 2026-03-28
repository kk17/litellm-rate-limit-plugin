"""Unit tests for ProviderProbeConfig."""

import pytest

from litellm_rate_limit.health_state import HealthStateManager
from litellm_rate_limit.provider_probe import ProviderProbeConfig, _extract_model_prefix


class TestExtractModelPrefix:
    def test_extract_prefix_with_slash(self):
        assert _extract_model_prefix("openai/gpt-4") == "openai"
        assert _extract_model_prefix("anthropic/claude-3-opus") == "anthropic"
        assert _extract_model_prefix("minimax/abab6.5-chat") == "minimax"

    def test_extract_prefix_no_slash(self):
        assert _extract_model_prefix("minimax-m2") == "minimax"
        assert _extract_model_prefix("glm-4.5-air") == "glm"
        assert _extract_model_prefix("gemini-3-flash-preview") == "gemini"

    def test_extract_prefix_single_word(self):
        assert _extract_model_prefix("unknown-model") == "unknown"
        assert _extract_model_prefix("model") == "model"


class TestProviderProbeConfig:
    def test_init_empty(self):
        config = ProviderProbeConfig()
        assert config.probe_models_by_provider == {}
        assert config._prefix_to_probe == {}
        assert config._explicit_models == set()

    def test_init_with_config(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        assert "minimax" in config._prefix_to_probe
        assert "glm" in config._prefix_to_probe

    def test_probe_model_map_single_model(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        assert config._prefix_to_probe["minimax"] == "minimax-m2"
        assert "minimax-m2" in config._explicit_models

    def test_probe_model_map_multiple_models(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        assert config._prefix_to_probe["glm"] == "glm-4.5-air"
        assert "glm-4.5-air" in config._explicit_models
        assert "glm-5" in config._explicit_models

    def test_get_effective_model_probe_model(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        assert config.get_effective_model("minimax-m2") == "minimax-m2"

    def test_get_effective_model_unlisted_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        assert config.get_effective_model("minimax-other") == "minimax-m2"
        assert config.get_effective_model("minimax/abab6.5-chat") == "minimax-m2"

    def test_get_effective_model_explicit_gets_own_status(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        assert config.get_effective_model("glm-4.5-air") == "glm-4.5-air"
        assert config.get_effective_model("glm-5") == "glm-5"

    def test_get_effective_model_unlisted_uses_first_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        assert config.get_effective_model("glm-other") == "glm-4.5-air"

    def test_get_effective_model_no_provider_config(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        assert config.get_effective_model("openai/gpt-4") == "openai/gpt-4"
        assert config.get_effective_model("unknown-model") == "unknown-model"

    def test_get_effective_model_with_provider_prefix(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "github-copilot": ["gemini-3-flash-preview", "gpt-4o"],
            }
        )
        assert config.get_effective_model("gemini-3-flash-preview") == "gemini-3-flash-preview"
        assert config.get_effective_model("gpt-4o") == "gpt-4o"
        assert config.get_effective_model("gemini-other") == "gemini-3-flash-preview"

    def test_is_probe_model(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        assert config.is_probe_model("minimax-m2") is True
        assert config.is_probe_model("minimax-other") is False

    def test_update_config(self):
        config = ProviderProbeConfig()
        config.update_config({"minimax": ["minimax-m2"]})
        assert "minimax" in config._prefix_to_probe


class TestHealthStateManagerWithProbeConfig:
    @pytest.mark.asyncio
    async def test_mark_rate_limited_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("minimax-other", 60.0)

        assert "minimax-m2" in manager._rate_limited_until
        assert "minimax-other" not in manager._rate_limited_until

    @pytest.mark.asyncio
    async def test_is_rate_limited_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        assert await manager.is_rate_limited("minimax-m2") is True
        assert await manager.is_rate_limited("minimax-other") is True
        assert await manager.is_rate_limited("minimax/abab6.5-chat") is True

    @pytest.mark.asyncio
    async def test_explicit_model_gets_own_status(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("glm-5", 60.0)

        assert await manager.is_rate_limited("glm-5") is True
        assert await manager.is_rate_limited("glm-4.5-air") is False
        assert await manager.is_rate_limited("glm-other") is False

    @pytest.mark.asyncio
    async def test_probe_model_blocks_others(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "zai": ["glm-4.5-air", "glm-5"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("glm-4.5-air", 60.0)

        assert await manager.is_rate_limited("glm-4.5-air") is True
        assert await manager.is_rate_limited("glm-5") is False
        assert await manager.is_rate_limited("glm-other") is True

    @pytest.mark.asyncio
    async def test_no_config_uses_original_model(self):
        manager = HealthStateManager()

        await manager.mark_rate_limited("minimax-m2", 60.0)

        assert await manager.is_rate_limited("minimax-m2") is True
        assert await manager.is_rate_limited("minimax-other") is False

    @pytest.mark.asyncio
    async def test_record_failure_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.record_failure("minimax-other", "Error")

        status = await manager.get_model_status("minimax-m2")
        assert status is not None
        assert status.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_record_success_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.record_success("minimax-other")

        status = await manager.get_model_status("minimax-m2")
        assert status is not None
        assert status.last_check_time is not None

    @pytest.mark.asyncio
    async def test_clear_rate_limit_uses_probe(self):
        config = ProviderProbeConfig(
            probe_models_by_provider={
                "minimax": ["minimax-m2"],
            }
        )
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        result = await manager.clear_rate_limit("minimax-other")
        assert result is True
        assert await manager.is_rate_limited("minimax-m2") is False
