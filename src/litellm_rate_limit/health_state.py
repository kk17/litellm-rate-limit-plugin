"""Health state manager for tracking rate-limited models."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm_rate_limit.alias_aware_state import AliasAwareHealthState
    from litellm_rate_limit.provider_probe import ProviderProbeConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthStatus:
    model_id: str
    is_rate_limited: bool = False
    rate_limited_until: float | None = None
    rate_limit_reset_at: datetime | None = None
    last_check_time: float | None = None
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass
class HealthStateManager:
    """Manages health state for models including rate limit tracking.

    Tracks which models are rate-limited and when they should be restored.
    Uses monotonic time for reliable duration tracking.

    When provider_probe_config is set, uses probe model health status
    for models under the same provider that are not explicitly listed.
    """

    provider_probe_config: "ProviderProbeConfig | None" = None
    _rate_limited_until: dict[str, float] = field(default_factory=dict)
    _rate_limit_reset_at: dict[str, datetime] = field(default_factory=dict)
    _model_status: dict[str, ModelHealthStatus] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())
    # Back-reference to AliasAwareHealthState for cross-state synchronization
    _alias_state: "AliasAwareHealthState | None" = None

    def _get_effective_model(self, model_id: str) -> str:
        if self.provider_probe_config is None:
            return model_id
        return self.provider_probe_config.get_effective_model(model_id)

    async def mark_rate_limited(
        self,
        model_id: str,
        seconds_until_reset: float,
        reset_at: datetime | None = None,
    ) -> None:
        effective_model = self._get_effective_model(model_id)
        until_monotonic = time.monotonic() + seconds_until_reset

        async with self._lock:
            self._rate_limited_until[effective_model] = until_monotonic
            if reset_at:
                self._rate_limit_reset_at[effective_model] = reset_at

            if effective_model in self._model_status:
                self._model_status[effective_model].is_rate_limited = True
                self._model_status[effective_model].rate_limited_until = until_monotonic
                self._model_status[effective_model].rate_limit_reset_at = reset_at
            else:
                self._model_status[effective_model] = ModelHealthStatus(
                    model_id=effective_model,
                    is_rate_limited=True,
                    rate_limited_until=until_monotonic,
                    rate_limit_reset_at=reset_at,
                )

        logger.info(
            "Model %s marked as rate-limited for %.1f seconds (resets at %s)",
            effective_model,
            seconds_until_reset,
            reset_at or "unknown",
        )

    async def is_rate_limited(self, model_id: str) -> bool:
        effective_model = self._get_effective_model(model_id)

        async with self._lock:
            if effective_model not in self._rate_limited_until:
                return False

            if time.monotonic() >= self._rate_limited_until[effective_model]:
                del self._rate_limited_until[effective_model]
                self._rate_limit_reset_at.pop(effective_model, None)

                if effective_model in self._model_status:
                    self._model_status[effective_model].is_rate_limited = False
                    self._model_status[effective_model].rate_limited_until = None
                    self._model_status[effective_model].rate_limit_reset_at = None

                logger.info("Model %s automatically restored after rate limit expiry", effective_model)
                return False

            return True

    def get_rate_limit_until(self, model_id: str) -> datetime | None:
        effective_model = self._get_effective_model(model_id)
        reset_at = self._rate_limit_reset_at.get(effective_model)
        return reset_at

    async def get_healthy_models(self, all_models: list[str]) -> list[str]:
        healthy = []
        for model in all_models:
            if not await self.is_rate_limited(model):
                healthy.append(model)
        return healthy

    async def get_rate_limited_models(self) -> dict[str, datetime]:
        result = {}
        now = time.monotonic()

        async with self._lock:
            expired = [k for k, v in self._rate_limited_until.items() if now >= v]
            for k in expired:
                del self._rate_limited_until[k]
                self._rate_limit_reset_at.pop(k, None)
                if k in self._model_status:
                    self._model_status[k].is_rate_limited = False

            for model_id, reset_at in self._rate_limit_reset_at.items():
                result[model_id] = reset_at

        return result

    async def clear_rate_limit(self, model_id: str) -> bool:
        effective_model = self._get_effective_model(model_id)

        async with self._lock:
            if effective_model in self._rate_limited_until:
                del self._rate_limited_until[effective_model]
                self._rate_limit_reset_at.pop(effective_model, None)

                if effective_model in self._model_status:
                    self._model_status[effective_model].is_rate_limited = False
                    self._model_status[effective_model].rate_limited_until = None
                    self._model_status[effective_model].rate_limit_reset_at = None

                logger.info("Manually cleared rate limit for model %s", effective_model)
                return True
            return False

    async def get_model_status(self, model_id: str) -> ModelHealthStatus | None:
        effective_model = self._get_effective_model(model_id)
        await self.is_rate_limited(effective_model)

        async with self._lock:
            return self._model_status.get(effective_model)

    async def record_failure(self, model_id: str, error: str) -> None:
        effective_model = self._get_effective_model(model_id)

        async with self._lock:
            if effective_model in self._model_status:
                self._model_status[effective_model].consecutive_failures += 1
                self._model_status[effective_model].last_error = error
            else:
                self._model_status[effective_model] = ModelHealthStatus(
                    model_id=effective_model,
                    consecutive_failures=1,
                    last_error=error,
                )

    async def record_success(self, model_id: str) -> None:
        effective_model = self._get_effective_model(model_id)

        async with self._lock:
            if effective_model in self._model_status:
                self._model_status[effective_model].consecutive_failures = 0
                self._model_status[effective_model].last_error = None
                self._model_status[effective_model].last_check_time = time.monotonic()
            else:
                self._model_status[effective_model] = ModelHealthStatus(
                    model_id=effective_model,
                    last_check_time=time.monotonic(),
                )

    def get_remaining_cooldown(self, model_id: str) -> float | None:
        """Return the remaining cooldown seconds for a model, or None if not rate-limited."""
        effective_model = self._get_effective_model(model_id)
        if effective_model not in self._rate_limited_until:
            return None
        remaining = self._rate_limited_until[effective_model] - time.monotonic()
        return max(0.0, remaining)

    async def clear_all(self) -> None:
        async with self._lock:
            self._rate_limited_until.clear()
            self._rate_limit_reset_at.clear()
            for status in self._model_status.values():
                status.is_rate_limited = False
                status.rate_limited_until = None
                status.rate_limit_reset_at = None
