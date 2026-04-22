"""Unit tests for ProviderProbeConfig."""

import pytest

from litellm_rate_limit.health_state import HealthStateManager
from litellm_rate_limit.provider_probe import ProviderProbeConfig


class TestProviderProbeConfig:
    """Tests for ProviderProbeConfig with router-based building."""

    def test_init_empty(self):
        config = ProviderProbeConfig()
        assert config.models_to_check == []
        assert not config.is_built()

    def test_init_with_config(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        assert config.models_to_check == [{"minimax-m2": ["minimax-m2", "minimax-other"]}]
        assert not config.is_built()

    def test_build_from_router_single_provider(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/Minimax-Other"}},
        ]
        config.build_from_router(model_list)

        assert config.is_built()
        assert config.is_probe_model("minimax-m2")
        assert config.get_effective_model("minimax-other") == "minimax-m2"
        assert config.get_effective_model("minimax-m2") == "minimax-m2"

    def test_build_from_router_multiple_probes(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {"glm-4.5-air": ["glm-4.5-air", "glm-4.6"]},
                {"glm-5": ["glm-5"]},
            ]
        )
        model_list = [
            {"model_name": "glm-4.5-air", "litellm_params": {"model": "zai/glm-4.5-air"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
            {"model_name": "glm-4.6", "litellm_params": {"model": "zai/glm-4.6"}},
        ]
        config.build_from_router(model_list)

        assert config.is_probe_model("glm-4.5-air")
        assert config.is_probe_model("glm-5")
        assert config.get_effective_model("glm-4.5-air") == "glm-4.5-air"
        assert config.get_effective_model("glm-5") == "glm-5"
        assert config.get_effective_model("glm-4.6") == "glm-4.5-air"

    def test_build_from_router_cross_probe_resolution(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {"minimax-m2": ["minimax-m2", "minimax-other"]},
                {"glm-4.5-air": ["glm-4.5-air", "glm-5"]},
            ]
        )
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/Minimax-Other"}},
            {"model_name": "glm-4.5-air", "litellm_params": {"model": "zai/glm-4.5-air"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
        ]
        config.build_from_router(model_list)

        assert config.get_effective_model("minimax-other") == "minimax-m2"
        assert config.get_effective_model("glm-5") == "glm-4.5-air"

    def test_get_effective_model_probe_model(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
        ]
        config.build_from_router(model_list)

        assert config.get_effective_model("minimax-m2") == "minimax-m2"

    def test_get_effective_model_independent_model(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {"glm-4.5-air": ["glm-4.5-air", "glm-4.6"]},
                {"glm-5": ["glm-5"]},
            ]
        )
        model_list = [
            {"model_name": "glm-4.5-air", "litellm_params": {"model": "zai/glm-4.5-air"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
        ]
        config.build_from_router(model_list)

        assert config.get_effective_model("glm-5") == "glm-5"

    def test_get_effective_model_unlisted_model(self):
        config = ProviderProbeConfig()
        assert config.get_effective_model("any-model") == "any-model"

    def test_is_probe_model(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
        ]
        config.build_from_router(model_list)

        assert config.is_probe_model("minimax-m2") is True
        assert config.is_probe_model("minimax-other") is False

    def test_update_config(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
        ]
        config.build_from_router(model_list)
        assert config.is_built()

        config.update_config([{"glm-4.5-air": ["glm-4.5-air"]}])
        assert not config.is_built()

    def test_get_probe_share_map(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/Minimax-Other"}},
        ]
        config.build_from_router(model_list)

        share_map = config.get_probe_share_map()
        assert "minimax-m2" in share_map
        assert "minimax-other" in share_map["minimax-m2"]

    def test_get_models_to_health_check(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/Minimax-Other"}},
        ]
        config.build_from_router(model_list)

        all_models = ["minimax-m2", "minimax-other"]
        to_check = config.get_models_to_health_check(all_models)

        assert "minimax-m2" in to_check
        assert "minimax-other" not in to_check


class TestHealthStateManagerWithProbeConfig:
    """Tests for HealthStateManager with ProviderProbeConfig."""

    @pytest.fixture
    def minimax_config(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/Minimax-Other"}},
        ]
        config.build_from_router(model_list)
        return config

    @pytest.fixture
    def zai_config(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {"glm-4.5-air": ["glm-4.5-air", "glm-4.6"]},
                {"glm-5": ["glm-5"]},
            ]
        )
        model_list = [
            {"model_name": "glm-4.5-air", "litellm_params": {"model": "zai/glm-4.5-air"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
            {"model_name": "glm-4.6", "litellm_params": {"model": "zai/glm-4.6"}},
        ]
        config.build_from_router(model_list)
        return config

    @pytest.mark.asyncio
    async def test_mark_rate_limited_uses_probe(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.mark_rate_limited("minimax-other", 60.0)

        assert await manager.is_rate_limited("minimax-m2") is True

    @pytest.mark.asyncio
    async def test_is_rate_limited_uses_probe(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        assert await manager.is_rate_limited("minimax-other") is True

    @pytest.mark.asyncio
    async def test_independent_probe_gets_own_status(self, zai_config):
        manager = HealthStateManager(provider_probe_config=zai_config)

        await manager.mark_rate_limited("glm-4.5-air", 60.0)

        assert await manager.is_rate_limited("glm-5") is False
        assert await manager.is_rate_limited("glm-4.6") is True

    @pytest.mark.asyncio
    async def test_probe_model_blocks_others(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        assert await manager.is_rate_limited("minimax-m2") is True
        assert await manager.is_rate_limited("minimax-other") is True

    @pytest.mark.asyncio
    async def test_no_config_uses_original_model(self):
        manager = HealthStateManager(provider_probe_config=None)

        await manager.mark_rate_limited("minimax-other", 60.0)

        assert await manager.is_rate_limited("minimax-other") is True
        assert await manager.is_rate_limited("minimax-m2") is False

    @pytest.mark.asyncio
    async def test_record_failure_uses_probe(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.record_failure("minimax-other", "Error")

        status = await manager.get_model_status("minimax-m2")
        assert status is not None
        assert status.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_record_success_uses_probe(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.record_success("minimax-other")

        status = await manager.get_model_status("minimax-m2")
        assert status is not None
        assert status.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_clear_rate_limit_uses_probe(self, minimax_config):
        manager = HealthStateManager(provider_probe_config=minimax_config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        result = await manager.clear_rate_limit("minimax-other")
        assert result is True
        assert await manager.is_rate_limited("minimax-m2") is False


class TestProviderProbeConfigLitellmModelMapping:
    """Test litellm_model mapping for health check API calls.

    Health checks need the provider-prefixed model string (litellm_model)
    while health state tracks by model_name.
    """

    def test_get_litellm_model(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/MiniMax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/MiniMax-Other"}},
        ]
        config.build_from_router(model_list)

        assert config.get_litellm_model("minimax-m2") == "minimax/MiniMax-M2"
        assert config.get_litellm_model("minimax-other") == "minimax/MiniMax-Other"
        assert config.get_litellm_model("nonexistent") is None

    def test_config_key_matches_litellm_suffix(self):
        """Config keys are matched against model_name (case-insensitive from litellm_model suffix)."""
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/MiniMax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/MiniMax-Other"}},
        ]
        config.build_from_router(model_list)

        assert config.is_probe_model("minimax-m2")
        assert config.get_effective_model("minimax-m2") == "minimax-m2"
        assert config.get_effective_model("minimax-other") == "minimax-m2"

    def test_get_models_to_health_check_filters_shared_models(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/MiniMax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/MiniMax-Other"}},
        ]
        config.build_from_router(model_list)

        all_models = ["minimax-m2", "minimax-other"]
        to_check = config.get_models_to_health_check(all_models)

        assert "minimax-m2" in to_check
        assert "minimax-other" not in to_check

    @pytest.mark.asyncio
    async def test_health_state_with_shared_probe(self):
        config = ProviderProbeConfig(models_to_check=[{"minimax-m2": ["minimax-m2", "minimax-other"]}])
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/MiniMax-M2"}},
            {"model_name": "minimax-other", "litellm_params": {"model": "minimax/MiniMax-Other"}},
        ]
        config.build_from_router(model_list)
        manager = HealthStateManager(provider_probe_config=config)

        await manager.mark_rate_limited("minimax-m2", 60.0)
        assert await manager.is_rate_limited("minimax-other") is True


class TestGithubCopilotScenario:
    """Test github_copilot provider with heterogeneous models.

    github_copilot has models like grok-code-fast-1, gpt-4o, gemini-3-flash-preview, etc.
    Each probe model shares health status with models in its list.
    Probe models can have independent status if they're their own list.
    """

    @pytest.fixture
    def github_copilot_config(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {
                    "grok-code-fast-1": [
                        "grok-code-fast-1",
                        "gpt-5.2",
                        "claude-sonnet-4.6",
                    ]
                },
                {"gemini-3-flash-preview": ["gemini-3-flash-preview"]},
                {"gpt-4o": ["gpt-4o"]},
                {"gpt-4.1": ["gpt-4.1"]},
                {"gpt-5-mini": ["gpt-5-mini"]},
            ]
        )
        model_list = [
            {
                "model_name": "grok-code-fast-1",
                "litellm_params": {"model": "github_copilot/grok-code-fast-1"},
            },
            {
                "model_name": "gemini-3-flash-preview",
                "litellm_params": {"model": "github_copilot/gemini-3-flash-preview"},
            },
            {"model_name": "gpt-4o", "litellm_params": {"model": "github_copilot/gpt-4o"}},
            {"model_name": "gpt-4.1", "litellm_params": {"model": "github_copilot/gpt-4.1"}},
            {"model_name": "gpt-5-mini", "litellm_params": {"model": "github_copilot/gpt-5-mini"}},
            {"model_name": "gpt-5.2", "litellm_params": {"model": "github_copilot/gpt-5.2"}},
            {
                "model_name": "claude-sonnet-4.6",
                "litellm_params": {"model": "github_copilot/claude-sonnet-4.6"},
            },
        ]
        config.build_from_router(model_list)
        return config

    def test_probe_model_is_grok(self, github_copilot_config):
        assert github_copilot_config.is_probe_model("grok-code-fast-1") is True

    def test_independent_models_have_own_status(self, github_copilot_config):
        assert github_copilot_config.is_probe_model("gemini-3-flash-preview") is True
        assert github_copilot_config.is_probe_model("gpt-4o") is True
        assert github_copilot_config.is_probe_model("gpt-4.1") is True
        assert github_copilot_config.is_probe_model("gpt-5-mini") is True

    def test_unlisted_model_uses_probe(self, github_copilot_config):
        assert github_copilot_config.get_effective_model("gpt-5.2") == "grok-code-fast-1"
        assert github_copilot_config.get_effective_model("claude-sonnet-4.6") == "grok-code-fast-1"

    def test_independent_model_returns_self(self, github_copilot_config):
        assert github_copilot_config.get_effective_model("gpt-4o") == "gpt-4o"
        assert github_copilot_config.get_effective_model("gemini-3-flash-preview") == "gemini-3-flash-preview"
        assert github_copilot_config.get_effective_model("grok-code-fast-1") == "grok-code-fast-1"

    @pytest.mark.asyncio
    async def test_health_status_shared_across_models(self, github_copilot_config):
        manager = HealthStateManager(provider_probe_config=github_copilot_config)

        await manager.mark_rate_limited("grok-code-fast-1", 60.0)

        assert await manager.is_rate_limited("gpt-5.2") is True
        assert await manager.is_rate_limited("claude-sonnet-4.6") is True
        assert await manager.is_rate_limited("gpt-4o") is False
        assert await manager.is_rate_limited("gemini-3-flash-preview") is False

    @pytest.mark.asyncio
    async def test_mark_rate_limited_on_shared_model(self, github_copilot_config):
        manager = HealthStateManager(provider_probe_config=github_copilot_config)

        await manager.mark_rate_limited("gpt-5.2", 60.0)

        assert await manager.is_rate_limited("grok-code-fast-1") is True
        assert await manager.is_rate_limited("claude-sonnet-4.6") is True

    def test_probe_share_map(self, github_copilot_config):
        share_map = github_copilot_config.get_probe_share_map()

        assert "grok-code-fast-1" in share_map
        assert "gpt-5.2" in share_map["grok-code-fast-1"]
        assert "claude-sonnet-4.6" in share_map["grok-code-fast-1"]
        assert "gpt-4o" not in share_map.get("grok-code-fast-1", [])

    def test_models_to_health_check(self, github_copilot_config):
        all_models = [
            "grok-code-fast-1",
            "gemini-3-flash-preview",
            "gpt-4o",
            "gpt-4.1",
            "gpt-5-mini",
            "gpt-5.2",
            "claude-sonnet-4.6",
        ]
        to_check = github_copilot_config.get_models_to_health_check(all_models)

        assert "grok-code-fast-1" in to_check
        assert "gemini-3-flash-preview" in to_check
        assert "gpt-4o" in to_check
        assert "gpt-4.1" in to_check
        assert "gpt-5-mini" in to_check
        assert "gpt-5.2" not in to_check
        assert "claude-sonnet-4.6" not in to_check


class TestMultiProviderScenario:
    """Test multiple probes with different configurations."""

    @pytest.fixture
    def multi_probe_config(self):
        config = ProviderProbeConfig(
            models_to_check=[
                {"minimax-m2": ["minimax-m2", "minimax-m2.5"]},
                {"glm-4.5-air": ["glm-4.5-air", "glm-5"]},
                {
                    "grok-code-fast-1": [
                        "grok-code-fast-1",
                        "gpt-5.2",
                    ]
                },
                {"gemini-3-flash-preview": ["gemini-3-flash-preview"]},
                {"gpt-4o": ["gpt-4o"]},
            ]
        )
        model_list = [
            {"model_name": "minimax-m2", "litellm_params": {"model": "minimax/Minimax-M2"}},
            {"model_name": "minimax-m2.5", "litellm_params": {"model": "minimax/Minimax-M2.5"}},
            {"model_name": "glm-4.5-air", "litellm_params": {"model": "zai/glm-4.5-air"}},
            {"model_name": "glm-5", "litellm_params": {"model": "zai/glm-5"}},
            {
                "model_name": "grok-code-fast-1",
                "litellm_params": {"model": "github_copilot/grok-code-fast-1"},
            },
            {"model_name": "gpt-4o", "litellm_params": {"model": "github_copilot/gpt-4o"}},
            {"model_name": "gpt-5.2", "litellm_params": {"model": "github_copilot/gpt-5.2"}},
        ]
        config.build_from_router(model_list)
        return config

    @pytest.mark.asyncio
    async def test_probes_independent(self, multi_probe_config):
        manager = HealthStateManager(provider_probe_config=multi_probe_config)

        await manager.mark_rate_limited("minimax-m2", 60.0)

        assert await manager.is_rate_limited("minimax-m2.5") is True
        assert await manager.is_rate_limited("glm-5") is False
        assert await manager.is_rate_limited("gpt-5.2") is False

    @pytest.mark.asyncio
    async def test_cross_probe_isolation(self, multi_probe_config):
        manager = HealthStateManager(provider_probe_config=multi_probe_config)

        await manager.mark_rate_limited("grok-code-fast-1", 60.0)

        assert await manager.is_rate_limited("gpt-5.2") is True
        assert await manager.is_rate_limited("gpt-4o") is False
        assert await manager.is_rate_limited("minimax-m2") is False
        assert await manager.is_rate_limited("glm-5") is False
