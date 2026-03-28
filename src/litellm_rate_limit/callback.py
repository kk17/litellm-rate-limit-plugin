"""Custom callback for LiteLLM to handle rate limits."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from litellm.integrations.custom_logger import CustomLogger

from litellm_rate_limit.parser import extract_rate_limit_reset_seconds, is_rate_limit_error

if TYPE_CHECKING:
    from litellm.proxy.types import UserAPIKeyAuth

    try:
        from litellm.router import Router as LiteLLMRouter
    except ImportError:
        LiteLLMRouter = None

logger = logging.getLogger(__name__)


class RateLimitCallback(CustomLogger):
    """LiteLLM callback that intercepts rate limit errors and sets precise cooldowns."""

    def __init__(
        self,
        default_cooldown_seconds: float = 60.0,
        router: Optional["LiteLLMRouter"] = None,
    ):
        self.default_cooldown_seconds = default_cooldown_seconds
        self._router = router
        self._cooldown_cache_lock = asyncio.Lock()

    def set_router(self, router: "LiteLLMRouter") -> None:
        self._router = router

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        traceback_str: str | None = None,
    ) -> None:
        if not is_rate_limit_error(original_exception):
            return

        model = request_data.get("model", "unknown")
        cooldown_seconds = extract_rate_limit_reset_seconds(
            original_exception,
            default=self.default_cooldown_seconds,
        )

        logger.info(
            "Rate limit detected for model %s, cooldown for %.1f seconds",
            model,
            cooldown_seconds,
        )

        if self._router is not None:
            await self._update_cooldown(model, cooldown_seconds)

    async def _update_cooldown(self, model: str, cooldown_seconds: float) -> None:
        if not hasattr(self._router, "cooldown_cache"):
            logger.warning("Router has no cooldown_cache attribute")
            return

        async with self._cooldown_cache_lock:
            deployment = self._get_deployment_for_model(model)
            if deployment is None:
                logger.warning("No deployment found for model %s", model)
                return

            cooldown_cache = self._router.cooldown_cache
            if hasattr(cooldown_cache, "set_cooldown"):
                await cooldown_cache.set_cooldown(
                    model_id=deployment,
                    cooldown_time=cooldown_seconds,
                )
                logger.debug("Set cooldown for deployment %s: %.1fs", deployment, cooldown_seconds)
            else:
                logger.warning("cooldown_cache has no set_cooldown method")

    def _get_deployment_for_model(self, model: str) -> str | None:
        if self._router is None:
            return None

        if hasattr(self._router, "get_deployment"):
            try:
                deployment = self._router.get_deployment(model_id=model)
                if deployment and hasattr(deployment, "model_info"):
                    return deployment.model_info.get("id", model)
            except Exception as e:
                logger.debug("Could not get deployment for model %s: %s", model, e)

        return model
