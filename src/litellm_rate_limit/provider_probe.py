"""Provider probe model mapping for health status sharing.

Allows specifying probe models per provider. The first model's health status
is shared with other models from the same provider that are not explicitly listed.

Example:
    probe_models_by_provider:
        minimax: ["minimax-m2"]
        zai: ["glm-4.5-air", "glm-5"]
        github_copilot: ["grok-code-fast-1", "gemini-3-flash-preview", "gpt-4o"]

    - minimax-m2 health status is used for ALL minimax models
    - glm-4.5-air health status is used for all zai models EXCEPT glm-5
    - glm-5 has its own independent health status
    - grok-code-fast-1 health status is used for all github_copilot models EXCEPT
      gemini-3-flash-preview and gpt-4o (which have their own status)

Data flow:
1. Router's model_list contains: {"model_name": "gpt-5.2", "litellm_params": {"model": "github_copilot/gpt-5.2"}}
2. Provider is extracted from litellm_params.model (e.g., "github_copilot/gpt-5.2" -> "github_copilot")
3. probe_models_by_provider maps provider -> [probe_model, explicit_models...]
4. build_from_router() creates model_name -> probe_model_name mapping
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _extract_provider_from_litellm_model(litellm_model: str) -> str:
    """Extract provider from litellm_params.model field.

    Examples:
        "github_copilot/gpt-5.2" -> "github_copilot"
        "minimax/Minimax-M2" -> "minimax"
        "zai/glm-5" -> "zai"
        "gpt-4" -> "" (no provider prefix)
    """
    if "/" in litellm_model:
        return litellm_model.split("/")[0]
    return ""


def _extract_model_prefix(model_id: str) -> str:
    """Extract prefix from model_id for fallback matching.

    Handles formats like:
    - "openai/gpt-4" -> "openai"
    - "anthropic/claude-3-opus" -> "anthropic"
    - "minimax-m2" -> "minimax" (no slash, use prefix before first hyphen)
    - "glm-4.5-air" -> "glm"
    """
    if "/" in model_id:
        return model_id.split("/")[0]

    parts = model_id.split("-")
    if len(parts) >= 2:
        return parts[0]

    return model_id


@dataclass
class ProviderProbeConfig:
    """Configuration for provider probe models."""

    probe_models_by_provider: dict[str, list[str]] = field(default_factory=dict)

    _provider_to_models: dict[str, set[str]] = field(default_factory=dict)
    _model_to_probe: dict[str, str] = field(default_factory=dict)
    _explicit_models: set[str] = field(default_factory=set)
    _probe_models: set[str] = field(default_factory=set)
    _is_built: bool = False
    _model_name_to_litellm_model: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def build_from_router(self, model_list: list[dict]) -> None:
        """Build caches from router's model_list.

        Config names in probe_models_by_provider are resolved against both
        model_name and the model suffix of litellm_model. This allows config
        entries like "MiniMax-M2" to match deployments where model_name is
        "minimax-m2" but litellm_params.model is "minimax/MiniMax-M2".

        Args:
            model_list: List of deployment dicts from router.model_list
                Each dict has:
                - model_name: The alias/short name (e.g., "gpt-5.2")
                - litellm_params.model: Full model path (e.g., "github_copilot/gpt-5.2")
        """
        self._provider_to_models.clear()
        self._model_to_probe.clear()
        self._explicit_models.clear()
        self._probe_models.clear()
        self._model_name_to_litellm_model.clear()

        if not self.probe_models_by_provider:
            self._is_built = True
            logger.debug("No probe_models_by_provider configured, skipping build")
            return

        provider_lookup: dict[str, dict[str, str]] = {}

        for deployment in model_list:
            model_name = (
                deployment.get("model_name")
                if isinstance(deployment, dict)
                else getattr(deployment, "model_name", None)
            )
            if not model_name:
                continue

            litellm_params = (
                deployment.get("litellm_params", {})
                if isinstance(deployment, dict)
                else getattr(deployment, "litellm_params", {})
            )
            litellm_model = (
                litellm_params.get("model", "")
                if isinstance(litellm_params, dict)
                else getattr(litellm_params, "model", "")
            )

            if litellm_model:
                self._model_name_to_litellm_model[model_name] = litellm_model

            provider = _extract_provider_from_litellm_model(litellm_model)
            if not provider:
                continue

            if provider not in self._provider_to_models:
                self._provider_to_models[provider] = set()
                provider_lookup[provider] = {}
            self._provider_to_models[provider].add(model_name)

            # Resolve config names against both model_name and litellm_model suffix
            # e.g., config "MiniMax-M2" matches litellm_model "minimax/MiniMax-M2" -> model_name "minimax-m2"
            provider_lookup[provider][model_name] = model_name

            if "/" in litellm_model:
                model_suffix = litellm_model.split("/", 1)[1]
                provider_lookup[provider][model_suffix] = model_name

        for provider, configured_models in self.probe_models_by_provider.items():
            if not configured_models:
                continue

            lookup = provider_lookup.get(provider, {})

            probe_config_name = configured_models[0]
            probe_model_name = lookup.get(probe_config_name, probe_config_name)

            explicit_config_names = configured_models[1:] if len(configured_models) > 1 else []
            explicit_model_names: set[str] = set()
            for config_name in explicit_config_names:
                resolved = lookup.get(config_name, config_name)
                explicit_model_names.add(resolved)

            self._probe_models.add(probe_model_name)
            self._explicit_models.update(explicit_model_names)

            provider_models = self._provider_to_models.get(provider, set())

            for model_name in provider_models:
                if model_name in explicit_model_names:
                    self._model_to_probe[model_name] = model_name
                elif model_name == probe_model_name:
                    self._model_to_probe[model_name] = probe_model_name
                else:
                    self._model_to_probe[model_name] = probe_model_name

        self._is_built = True
        logger.debug(
            "Built provider probe caches: provider_to_models=%s, model_to_probe=%s, explicit=%s, probes=%s, litellm_models=%s",
            {k: list(v) for k, v in self._provider_to_models.items()},
            self._model_to_probe,
            self._explicit_models,
            self._probe_models,
            self._model_name_to_litellm_model,
        )

    def update_config(self, probe_models_by_provider: dict[str, list[str]]) -> None:
        """Update probe configuration. Requires build_from_router() call after."""
        self.probe_models_by_provider = probe_models_by_provider
        self._is_built = False

    def is_built(self) -> bool:
        """Check if caches have been built from router."""
        return self._is_built

    def get_effective_model(self, model_id: str) -> str:
        """Get the effective model for health status lookup.

        Returns the probe model if this model should share its status,
        or the model itself if it has its own status.
        """
        if model_id in self._model_to_probe:
            return self._model_to_probe[model_id]

        if model_id in self._explicit_models:
            return model_id

        return model_id

    def is_probe_model(self, model_id: str) -> bool:
        """Check if this model is a probe model for its provider."""
        return model_id in self._probe_models

    def get_litellm_model(self, model_name: str) -> str | None:
        """Get the litellm_model string (with provider prefix) for a model_name."""
        return self._model_name_to_litellm_model.get(model_name)

    def is_explicit_model(self, model_id: str) -> bool:
        """Check if this model has its own health status (not shared)."""
        return model_id in self._explicit_models

    def get_explicit_models(self) -> set[str]:
        """Get all models that have their own health status."""
        return self._explicit_models.copy()

    def get_probe_models(self) -> set[str]:
        """Get all probe models (first model in each provider's list)."""
        return self._probe_models.copy()

    def get_models_to_health_check(self, all_router_models: list[str]) -> list[str]:
        """Get the list of models that need health checks.

        Only probe models and explicit models need health checks.
        Other models share the probe model's status.
        """
        if not self.probe_models_by_provider or not self._is_built:
            return list(all_router_models)

        models_to_check = self._probe_models | self._explicit_models

        for model in all_router_models:
            if model not in self._model_to_probe:
                models_to_check.add(model)

        result = sorted(models_to_check)
        logger.debug(
            "Health check: %d models to check (probes: %d, explicit: %d, uncovered: %d)",
            len(result),
            len(self._probe_models),
            len(self._explicit_models),
            len(models_to_check) - len(self._probe_models) - len(self._explicit_models),
        )
        return result

    def get_probe_share_map(self) -> dict[str, list[str]]:
        """Get the probe model -> affected models mapping.

        Returns:
            Dict mapping probe model to list of models that share its status.
        """
        share_map: dict[str, list[str]] = {}
        for model, probe in self._model_to_probe.items():
            if probe not in share_map:
                share_map[probe] = []
            if model != probe:
                share_map[probe].append(model)
        return share_map
