"""Custom callback for LiteLLM to handle rate limits."""

import asyncio
import logging
from typing import TYPE_CHECKING

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import DualCache

from litellm_rate_limit.alias_aware_state import AliasAwareHealthState
from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner
from litellm_rate_limit.health_state import HealthStateManager
from litellm_rate_limit.parser import extract_rate_limit_reset_seconds, is_rate_limit_error
from litellm_rate_limit.provider_probe import ProviderProbeConfig

if TYPE_CHECKING:
    from litellm.router import Router as LiteLLMRouter

logger = logging.getLogger(__name__)


class RateLimitCallback(CustomLogger):
    """LiteLLM callback that intercepts rate limit errors and blocks rate-limited models.

    Features:
    - Pre-call hook to skip rate-limited models
    - Post-failure hook to detect 429 errors and extract reset times
    - Optional background health checking
    - Provider probe model support for shared health status
    """

    def __init__(
        self,
        default_cooldown_seconds: float = 60.0,
        probe_models_by_provider: dict[str, list[str]] | None = None,
        health_check_enabled: bool = False,
        health_check_interval_seconds: int = 60,
        health_check_prompt: str = "Say 'ok'",
        health_check_max_latency_ms: float = 30000.0,
    ):
        self.default_cooldown_seconds = default_cooldown_seconds
        self._router: LiteLLMRouter | None = None
        self._cooldown_cache_lock = asyncio.Lock()

        self._probe_config: ProviderProbeConfig | None = None
        if probe_models_by_provider:
            self._probe_config = ProviderProbeConfig(probe_models_by_provider=probe_models_by_provider)

        self._health_state = HealthStateManager(provider_probe_config=self._probe_config)
        self._alias_state = AliasAwareHealthState()

        self._health_check_enabled = health_check_enabled
        self._health_check_interval = health_check_interval_seconds
        self._health_runner: HealthCheckRunner | None = None

        if health_check_enabled:
            benchmark = HealthBenchmark(
                test_prompt=health_check_prompt,
                max_latency_ms=health_check_max_latency_ms,
            )
            self._health_runner = HealthCheckRunner(benchmark=benchmark)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(
            "RateLimitCallback initialized: cooldown=%.1fs, health_check=%s, probe_config=%s",
            default_cooldown_seconds,
            health_check_enabled,
            bool(probe_models_by_provider),
        )

    def set_router(self, router: "LiteLLMRouter") -> None:
        """Set the router reference for cooldown cache access."""
        self._router = router
        self._alias_state.set_router(router)
        logger.info("Router reference set")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> dict:
        model = data.get("model", "")
        logger.debug("Pre-call hook for model %s", model)
        return data

        is_limited = await self._alias_state.is_rate_limited(model)
        if is_limited:
            logger.warning("Model %s is rate-limited, rejecting request", model)
            from litellm.exceptions import RejectedRequestError

            return RejectedRequestError(
                message=f"Model {model} is rate-limited",
                model=model,
                llm_provider="",
                request_data=data,
            )

        logger.debug("Pre-call check passed for model %s", model)
        return data

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        traceback_str: str | None = None,
    ) -> None:
        """Post-failure hook to detect rate limit errors and extract reset times."""
        logger.debug(
            "Post-failure hook called: exception=%s",
            type(original_exception).__name__,
        )

        if not is_rate_limit_error(original_exception):
            logger.debug("Not a rate limit error, skipping")
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

        await self._alias_state.mark_rate_limited(model, cooldown_seconds)

        if self._router is not None:
            await self._update_cooldown(model, cooldown_seconds)

    async def _update_cooldown(self, model: str, cooldown_seconds: float) -> None:
        """Update LiteLLM's cooldown cache."""
        if not hasattr(self._router, "cooldown_cache"):
            logger.debug("Router has no cooldown_cache attribute")
            return

        async with self._cooldown_cache_lock:
            deployment = self._get_deployment_for_model(model)
            if deployment is None:
                deployment = model

            cooldown_cache = self._router.cooldown_cache
            cooldown_cache.add_deployment_to_cooldown(
                model_id=deployment,
                original_exception=Exception("Rate limit detected"),
                exception_status=429,
                cooldown_time=cooldown_seconds,
            )
            logger.info("Set cooldown for deployment %s: %.1fs", deployment, cooldown_seconds)

    def _get_deployment_for_model(self, model: str) -> str | None:
        """Get the deployment ID for a model name."""
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

    async def start_health_checks(self, models: list[str]) -> None:
        """Start background health checking for the given models."""
        if not self._health_runner:
            logger.warning("Health checker not enabled")
            return

        await self._health_runner.start_periodic_checks(
            name="default",
            models=models,
            interval_seconds=self._health_check_interval,
            health_manager=self._health_state,
        )
        logger.info("Started health checks for %d models", len(models))

    async def stop_health_checks(self) -> None:
        """Stop all background health checks."""
        if self._health_runner:
            await self._health_runner.stop_all()
            logger.info("Stopped health checks")

    @property
    def health_state(self) -> HealthStateManager:
        """Get the health state manager for external access."""
        return self._health_state
