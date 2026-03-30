"""Custom callback for LiteLLM to handle rate limits and unhealthy models."""

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
from litellm_rate_limit.provider_probe import ProviderProbeConfig, _extract_model_prefix

if TYPE_CHECKING:
    from litellm.router import Router as LiteLLMRouter

logger = logging.getLogger(__name__)


class RateLimitCallback(CustomLogger):
    """LiteLLM callback that intercepts API errors and blocks unhealthy models.

    Handles 429 rate limits with header-parsed reset times, and all other API
    errors (except 401/403) with per-provider or default cooldown.
    """

    _SKIP_STATUS_CODES = {401, 403}

    def __init__(
        self,
        default_cooldown_seconds: float = 60.0,
        probe_models_by_provider: dict[str, list[str]] | None = None,
        provider_cooldown_seconds: dict[str, float] | None = None,
        health_check_enabled: bool = False,
        health_check_interval_seconds: int = 60,
        health_check_prompt: str = "Say 'ok'",
        health_check_max_latency_ms: float = 30000.0,
    ):
        self.default_cooldown_seconds = default_cooldown_seconds
        self.provider_cooldown_seconds = provider_cooldown_seconds or {}
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
            "RateLimitCallback initialized: cooldown=%.1fs, provider_cooldown=%s, health_check=%s, probe_config=%s",
            default_cooldown_seconds,
            bool(self.provider_cooldown_seconds),
            health_check_enabled,
            bool(probe_models_by_provider),
        )

    def set_router(self, router: "LiteLLMRouter") -> None:
        self._router = router
        self._alias_state.set_router(router)
        logger.info("Router reference set")

        if self._health_check_enabled and self._health_runner:
            model_names = self._get_model_names_from_router()
            if model_names:
                asyncio.create_task(
                    self._health_runner.start_periodic_checks(
                        name="startup",
                        models=model_names,
                        interval_seconds=self._health_check_interval,
                        health_manager=self._health_state,
                    )
                )
                logger.info("Started startup health checks for %d models", len(model_names))

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

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        traceback_str: str | None = None,
    ) -> None:
        logger.debug(
            "Post-failure hook called: exception=%s",
            type(original_exception).__name__,
        )

        status_code = self._get_status_code(original_exception)

        if status_code in self._SKIP_STATUS_CODES:
            logger.debug("Auth/permission error (%s), skipping", status_code)
            return

        if status_code is None:
            logger.debug("No status code on exception, skipping")
            return

        model = request_data.get("model", "unknown")

        if is_rate_limit_error(original_exception):
            cooldown_seconds = extract_rate_limit_reset_seconds(
                original_exception,
                default=self._get_cooldown_for_model(model),
            )
        else:
            cooldown_seconds = self._get_cooldown_for_model(model)

        logger.info(
            "Marking model %s unhealthy for %.1f seconds (status=%s)",
            model,
            cooldown_seconds,
            status_code,
        )

        await self._alias_state.mark_rate_limited(model, cooldown_seconds)

        if self._router is not None:
            await self._update_cooldown(model, cooldown_seconds, request_data)

    @staticmethod
    def _get_status_code(error: Exception) -> int | None:
        if hasattr(error, "status_code") and isinstance(error.status_code, int):
            return error.status_code
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            return error.response.status_code
        return None

    def _get_cooldown_for_model(self, model: str) -> float:
        prefix = _extract_model_prefix(model)
        if prefix in self.provider_cooldown_seconds:
            return self.provider_cooldown_seconds[prefix]
        return self.default_cooldown_seconds

    async def _update_cooldown(self, model: str, cooldown_seconds: float, request_data: dict) -> None:
        if not hasattr(self._router, "cooldown_cache"):
            logger.debug("Router has no cooldown_cache attribute")
            return

        async with self._cooldown_cache_lock:
            deployment_id = request_data.get("litellm_params", {}).get("model_info", {}).get("id")

            if deployment_id:
                self._router.cooldown_cache.add_deployment_to_cooldown(
                    model_id=deployment_id,
                    original_exception=Exception("Error detected by rate limit plugin"),
                    exception_status=429,
                    cooldown_time=cooldown_seconds,
                )
                logger.info(
                    "Set cooldown for deployment %s (model %s): %.1fs",
                    deployment_id,
                    model,
                    cooldown_seconds,
                )
                return

            deployment_ids = self._get_deployment_ids_for_model(model)
            if deployment_ids:
                for dep_id in deployment_ids:
                    self._router.cooldown_cache.add_deployment_to_cooldown(
                        model_id=dep_id,
                        original_exception=Exception("Error detected by rate limit plugin"),
                        exception_status=429,
                        cooldown_time=cooldown_seconds,
                    )
                    logger.info(
                        "Set cooldown for deployment %s (model %s): %.1fs",
                        dep_id,
                        model,
                        cooldown_seconds,
                    )
                return

            logger.warning(
                "No deployment ID found for model %s, using model name as fallback",
                model,
            )
            self._router.cooldown_cache.add_deployment_to_cooldown(
                model_id=model,
                original_exception=Exception("Error detected by rate limit plugin"),
                exception_status=429,
                cooldown_time=cooldown_seconds,
            )
            logger.info(
                "Set cooldown for model %s (fallback): %.1fs",
                model,
                cooldown_seconds,
            )

    def _get_deployment_ids_for_model(self, model_name: str) -> list[str]:
        if not hasattr(self._router, "model_list"):
            return []

        ids = []
        for deployment in self._router.model_list:
            dep_model_name = (
                deployment.get("model_name")
                if isinstance(deployment, dict)
                else getattr(deployment, "model_name", None)
            )
            if dep_model_name == model_name:
                model_info = (
                    deployment.get("model_info", {})
                    if isinstance(deployment, dict)
                    else getattr(deployment, "model_info", {})
                )
                dep_id = (
                    model_info.get("id") if isinstance(model_info, dict) else getattr(model_info, "id", None)
                )
                if dep_id:
                    ids.append(dep_id)
        return ids

    def _get_model_names_from_router(self) -> list[str]:
        if not hasattr(self._router, "model_list"):
            return []

        names = set()
        for deployment in self._router.model_list:
            model_name = (
                deployment.get("model_name")
                if isinstance(deployment, dict)
                else getattr(deployment, "model_name", None)
            )
            if model_name:
                names.add(model_name)
        return sorted(names)

    async def start_health_checks(self, models: list[str]) -> None:
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
        if self._health_runner:
            await self._health_runner.stop_all()
            logger.info("Stopped health checks")

    @property
    def health_state(self) -> HealthStateManager:
        return self._health_state
