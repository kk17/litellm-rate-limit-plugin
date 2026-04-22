"""Model probe mapping for health status sharing.

Allows specifying which models share health status via probe models.

Example:
    models_to_check:
      - minimax-m2.7:
        - minimax-m2.5
        - minimax-m2
      - glm-4.5-air:
        - glm-4.5-air
        - glm-4.5
        - glm-4.6
        - glm-4.7

    - minimax-m2.7 is the probe model; minimax-m2.5 and minimax-m2 share its status
    - glm-4.5-air is the probe model; glm-4.5, glm-4.6, glm-4.7 share its status
    - Models not mentioned in models_to_check are NOT health-checked (status: unknown)

Data flow:
1. Router's model_list contains: {"model_name": "gpt-5.2", "litellm_params": {"model": "github_copilot/gpt-5.2"}}
2. models_to_check is a list of dicts: {probe_model: [models_sharing_status...]}
3. build_from_router() creates model_name -> probe_model_name mapping for covered models
4. Health check uses litellm_model string from model_name -> litellm_model mapping
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderProbeConfig:
    """Configuration for model probe models with explicit sharing mappings."""

    models_to_check: list[dict[str, list[str]]] = field(default_factory=list)

    _model_to_probe: dict[str, str] = field(default_factory=dict)
    _probe_models: set[str] = field(default_factory=set)
    _is_built: bool = False
    _model_name_to_litellm_model: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def build_from_router(self, model_list: list[dict]) -> None:
        """Build caches from router's model_list.

        Parses models_to_check configuration to build:
        - model_name -> probe_model_name mapping for covered models
        - model_name -> litellm_model string mapping for API calls

        Models NOT mentioned in models_to_check are not added to _model_to_probe,
        which means they get their own independent health status.

        Args:
            model_list: List of deployment dicts from router.model_list
                Each dict has:
                - model_name: The alias/short name (e.g., "gpt-5.2")
                - litellm_params.model: Full model path (e.g., "github_copilot/gpt-5.2")
        """
        self._model_to_probe.clear()
        self._probe_models.clear()
        self._model_name_to_litellm_model.clear()

        if not self.models_to_check:
            self._is_built = True
            logger.debug("No models_to_check configured, skipping build")
            return

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

        for entry in self.models_to_check:
            if not isinstance(entry, dict):
                continue
            if len(entry) != 1:
                continue

            probe_model = list(entry.keys())[0]
            sharing_models = entry[probe_model]

            self._probe_models.add(probe_model)

            for model in sharing_models:
                self._model_to_probe[model] = probe_model

        self._is_built = True
        logger.debug(
            "Built probe config: model_to_probe=%s, probes=%s, litellm_models=%s",
            self._model_to_probe,
            self._probe_models,
            self._model_name_to_litellm_model,
        )

    def update_config(self, models_to_check: list[dict[str, list[str]]]) -> None:
        """Update probe configuration. Requires build_from_router() call after."""
        self.models_to_check = models_to_check
        self._is_built = False

    def is_built(self) -> bool:
        """Check if caches have been built from router."""
        return self._is_built

    def is_covered(self, model_id: str) -> bool:
        """Check if a model is covered by models_to_check config."""
        return model_id in self._model_to_probe

    def get_effective_model(self, model_id: str) -> str:
        """Get the effective model for health status lookup.

        Returns the probe model if this model shares its status,
        or the model itself if not covered by any probe config.
        """
        return self._model_to_probe.get(model_id, model_id)

    def is_probe_model(self, model_id: str) -> bool:
        """Check if this model is a probe model."""
        return model_id in self._probe_models

    def get_litellm_model(self, model_name: str) -> str | None:
        """Get the litellm_model string (with provider prefix) for a model_name."""
        return self._model_name_to_litellm_model.get(model_name)

    def get_models_to_health_check(self, all_router_models: list[str]) -> list[str]:
        """Get the list of models that need health checks.

        Only returns probe models (keys in models_to_check).
        Models not defined in models_to_check are NOT health-checked (status: unknown).
        """
        if not self.models_to_check or not self._is_built:
            return list(all_router_models)

        result = sorted(self._probe_models)
        logger.debug(
            "Health check: %d probe models to check (total router models: %d)",
            len(result),
            len(all_router_models),
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
