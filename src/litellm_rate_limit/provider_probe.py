"""Provider probe model mapping for health status sharing.

Allows specifying probe models per provider. The first model's health status
is shared with other models from the same provider that are not explicitly listed.

Example:
    probe_models_by_provider:
        minimax: ["minimax-m2"]
        zai: ["glm-4.5-air", "glm-5"]
        github-copilot: ["gemini-3-flash-preview", "gpt-4o", "gpt-4.1", "gpt-5-mini"]

    - minimax-m2 health status is used for ALL minimax models
    - glm-4.5-air health status is used for all zai models EXCEPT glm-5
    - glm-5 has its own independent health status
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _extract_model_prefix(model_id: str) -> str:
    """Extract prefix from model_id for matching.

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

    _prefix_to_probe: dict[str, str] = field(default_factory=dict)
    _explicit_models: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._build_caches()

    def _build_caches(self) -> None:
        self._prefix_to_probe.clear()
        self._explicit_models.clear()

        for _provider, models in self.probe_models_by_provider.items():
            if not models:
                continue

            probe_model = models[0]
            probe_prefix = _extract_model_prefix(probe_model)

            self._prefix_to_probe[probe_prefix] = probe_model

            for model in models:
                self._explicit_models.add(model)
                model_prefix = _extract_model_prefix(model)
                if "/" not in model:
                    self._explicit_models.add(f"{model_prefix}/{model}")

        logger.debug(
            "Built prefix-to-probe map: %s, explicit models: %s",
            self._prefix_to_probe,
            self._explicit_models,
        )

    def update_config(self, probe_models_by_provider: dict[str, list[str]]) -> None:
        self.probe_models_by_provider = probe_models_by_provider
        self._build_caches()

    def get_effective_model(self, model_id: str) -> str:
        if model_id in self._explicit_models:
            return model_id

        prefix = _extract_model_prefix(model_id)

        if prefix in self._prefix_to_probe:
            probe_model = self._prefix_to_probe[prefix]
            logger.debug(
                "Using probe model %s for model %s (prefix: %s)",
                probe_model,
                model_id,
                prefix,
            )
            return probe_model

        return model_id

    def is_probe_model(self, model_id: str) -> bool:
        prefix = _extract_model_prefix(model_id)
        if prefix in self._prefix_to_probe:
            return self._prefix_to_probe[prefix] == model_id
        return False

    def get_explicit_models(self) -> set[str]:
        return self._explicit_models.copy()

    def get_probe_models(self) -> dict[str, str]:
        return self._prefix_to_probe.copy()
