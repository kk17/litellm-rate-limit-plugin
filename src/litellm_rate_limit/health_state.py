"""Health state manager for tracking rate-limited models."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthStatus:
    model_id: str
    is_rate_limited: bool = False
    rate_limited_until: Optional[float] = None
    rate_limit_reset_at: Optional[datetime] = None
    last_check_time: Optional[float] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class HealthStateManager:
    """Manages health state for models including rate limit tracking.

    Tracks which models are rate-limited and when they should be restored.
    Uses monotonic time for reliable duration tracking.
    """

    _rate_limited_until: Dict[str, float] = field(default_factory=dict)
    _rate_limit_reset_at: Dict[str, datetime] = field(default_factory=dict)
    _model_status: Dict[str, ModelHealthStatus] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def mark_rate_limited(
        self,
        model_id: str,
        seconds_until_reset: float,
        reset_at: Optional[datetime] = None,
    ) -> None:
        until_monotonic = time.monotonic() + seconds_until_reset

        async with self._lock:
            self._rate_limited_until[model_id] = until_monotonic
            if reset_at:
                self._rate_limit_reset_at[model_id] = reset_at

            if model_id in self._model_status:
                self._model_status[model_id].is_rate_limited = True
                self._model_status[model_id].rate_limited_until = until_monotonic
                self._model_status[model_id].rate_limit_reset_at = reset_at
            else:
                self._model_status[model_id] = ModelHealthStatus(
                    model_id=model_id,
                    is_rate_limited=True,
                    rate_limited_until=until_monotonic,
                    rate_limit_reset_at=reset_at,
                )

        logger.info(
            "Model %s marked as rate-limited for %.1f seconds (resets at %s)",
            model_id,
            seconds_until_reset,
            reset_at or "unknown",
        )

    async def is_rate_limited(self, model_id: str) -> bool:
        async with self._lock:
            if model_id not in self._rate_limited_until:
                return False

            if time.monotonic() >= self._rate_limited_until[model_id]:
                del self._rate_limited_until[model_id]
                self._rate_limit_reset_at.pop(model_id, None)

                if model_id in self._model_status:
                    self._model_status[model_id].is_rate_limited = False
                    self._model_status[model_id].rate_limited_until = None
                    self._model_status[model_id].rate_limit_reset_at = None

                logger.info("Model %s automatically restored after rate limit expiry", model_id)
                return False

            return True

    async def get_healthy_models(self, all_models: List[str]) -> List[str]:
        healthy = []
        for model in all_models:
            if not await self.is_rate_limited(model):
                healthy.append(model)
        return healthy

    async def get_rate_limited_models(self) -> Dict[str, datetime]:
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
        async with self._lock:
            if model_id in self._rate_limited_until:
                del self._rate_limited_until[model_id]
                self._rate_limit_reset_at.pop(model_id, None)

                if model_id in self._model_status:
                    self._model_status[model_id].is_rate_limited = False
                    self._model_status[model_id].rate_limited_until = None
                    self._model_status[model_id].rate_limit_reset_at = None

                logger.info("Manually cleared rate limit for model %s", model_id)
                return True
            return False

    async def get_model_status(self, model_id: str) -> Optional[ModelHealthStatus]:
        await self.is_rate_limited(model_id)

        async with self._lock:
            return self._model_status.get(model_id)

    async def record_failure(self, model_id: str, error: str) -> None:
        async with self._lock:
            if model_id in self._model_status:
                self._model_status[model_id].consecutive_failures += 1
                self._model_status[model_id].last_error = error
            else:
                self._model_status[model_id] = ModelHealthStatus(
                    model_id=model_id,
                    consecutive_failures=1,
                    last_error=error,
                )

    async def record_success(self, model_id: str) -> None:
        async with self._lock:
            if model_id in self._model_status:
                self._model_status[model_id].consecutive_failures = 0
                self._model_status[model_id].last_error = None
                self._model_status[model_id].last_check_time = time.monotonic()
            else:
                self._model_status[model_id] = ModelHealthStatus(
                    model_id=model_id,
                    last_check_time=time.monotonic(),
                )

    async def clear_all(self) -> None:
        async with self._lock:
            self._rate_limited_until.clear()
            self._rate_limit_reset_at.clear()
            for status in self._model_status.values():
                status.is_rate_limited = False
                status.rate_limited_until = None
                status.rate_limit_reset_at = None
