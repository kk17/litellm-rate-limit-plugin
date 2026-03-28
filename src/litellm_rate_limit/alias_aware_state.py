"""Alias-aware health state tracking.

Integrates rate limit tracking with LiteLLM's model_group_alias feature.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    try:
        from litellm.router import Router as LiteLLMRouter
    except ImportError:
        LiteLLMRouter = None


@dataclass
class RateLimitEntry:
    model_id: str
    until_monotonic: float
    reset_at: Optional[datetime] = None


@dataclass
class AliasAwareHealthState:
    """Tracks rate limit state with alias resolution.

    When a target model is rate-limited, all its aliases are also blocked.
    """

    router: Optional["LiteLLMRouter"] = None
    _rate_limited_targets: Dict[str, RateLimitEntry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def set_router(self, router: "LiteLLMRouter") -> None:
        self.router = router

    def _resolve_to_target(self, model_name: str) -> str:
        if self.router is None:
            return model_name

        if hasattr(self.router, "model_group_alias"):
            return self.router.model_group_alias.get(model_name, model_name)

        return model_name

    async def mark_rate_limited(
        self,
        model_name: str,
        seconds_until_reset: float,
        reset_at: Optional[datetime] = None,
    ) -> None:
        target = self._resolve_to_target(model_name)
        until_monotonic = time.monotonic() + seconds_until_reset

        async with self._lock:
            self._rate_limited_targets[target] = RateLimitEntry(
                model_id=target,
                until_monotonic=until_monotonic,
                reset_at=reset_at,
            )

    async def is_rate_limited(self, model_name: str) -> bool:
        target = self._resolve_to_target(model_name)

        async with self._lock:
            if target not in self._rate_limited_targets:
                return False

            entry = self._rate_limited_targets[target]
            if time.monotonic() >= entry.until_monotonic:
                del self._rate_limited_targets[target]
                return False

            return True

    async def get_healthy_models(self, all_models: List[str]) -> List[str]:
        healthy = []
        for model in all_models:
            if not await self.is_rate_limited(model):
                healthy.append(model)
        return healthy

    async def clear_rate_limit(self, model_name: str) -> bool:
        target = self._resolve_to_target(model_name)

        async with self._lock:
            if target in self._rate_limited_targets:
                del self._rate_limited_targets[target]
                return True
            return False

    async def get_all_rate_limited(self) -> Dict[str, RateLimitEntry]:
        now = time.monotonic()
        async with self._lock:
            expired = [k for k, v in self._rate_limited_targets.items() if now >= v.until_monotonic]
            for k in expired:
                del self._rate_limited_targets[k]
            return dict(self._rate_limited_targets)

    async def clear_all(self) -> None:
        async with self._lock:
            self._rate_limited_targets.clear()
