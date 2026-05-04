"""Custom callback for LiteLLM to handle rate limits and unhealthy models."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.utils import DualCache

from litellm_rate_limit.alias_aware_state import AliasAwareHealthState
from litellm_rate_limit.config import RateLimitPluginConfig
from litellm_rate_limit.health_checker import HealthBenchmark, HealthCheckRunner
from litellm_rate_limit.health_state import HealthStateManager
from litellm_rate_limit.parser import detect_api_error, extract_rate_limit_reset_seconds, is_rate_limit_error
from litellm_rate_limit.provider_probe import ProviderProbeConfig

if TYPE_CHECKING:
    from litellm.router import Router as LiteLLMRouter

logger = logging.getLogger(__name__)


class RateLimitCallback(CustomLogger):
    """LiteLLM callback that intercepts API errors and blocks unhealthy models.

    Handles 429 rate limits with header-parsed reset times and all other API
    errors (except 401/403) with per-provider or default cooldown.
    """

    _SKIP_STATUS_CODES = {401, 403}

    def __init__(
        self,
        default_cooldown_seconds: float = 60.0,
        models_to_check: list[dict[str, list[str]]] | None = None,
        health_check_enabled: bool = False,
        health_check_interval_seconds: int = 60,
        health_check_prompt: str = "Say 'ok'",
        health_check_max_latency_ms: float = 30000.0,
        log_level: str = "DEBUG",
        litellm_log_level: str = "INFO",
        litellm_proxy_log_level: str = "INFO",
        litellm_router_log_level: str = "INFO",
    ):
        self.default_cooldown_seconds = default_cooldown_seconds
        self._router: LiteLLMRouter | None = None
        self._cooldown_cache_lock = asyncio.Lock()

        self._probe_config: ProviderProbeConfig | None = None
        if models_to_check:
            self._probe_config = ProviderProbeConfig(models_to_check=models_to_check)

        self._health_state = HealthStateManager(provider_probe_config=self._probe_config)
        self._alias_state = AliasAwareHealthState()
        # Wire alias_state back-reference for cross-state synchronization
        self._health_state.set_alias_state(self._alias_state)
        self._model_name_to_litellm_model: dict[str, str] = {}

        self._health_check_enabled = health_check_enabled
        self._health_check_interval = health_check_interval_seconds
        self._health_runner: HealthCheckRunner | None = None
        self._health_checks_started = False
        self._startup_models: list[str] | None = None
        self._startup_thread: threading.Thread | None = None
        self._health_check_loop: asyncio.AbstractEventLoop | None = None

        health_logger = logging.getLogger("litellm_rate_limit.health_checker")
        provider_logger = logging.getLogger("litellm_rate_limit.provider_probe")
        config_logger = logging.getLogger("litellm_rate_limit.config")
        state_logger = logging.getLogger("litellm_rate_limit.health_state")

        plugin_logger = logging.getLogger("litellm_rate_limit")
        callback_logger = logging.getLogger("litellm_rate_limit.callback")
        plugin_loggers = [
            plugin_logger,
            callback_logger,
            health_logger,
            provider_logger,
            config_logger,
            state_logger,
        ]

        litellm_format = (
            "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s"
        )
        formatter = logging.Formatter(litellm_format, datefmt="%Y-%m-%d %H:%M:%S")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        for lg in plugin_loggers:
            lg.handlers.clear()
            lg.propagate = False
            lg.addHandler(handler)
            lg.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        litellm_main_logger = logging.getLogger("LiteLLM")
        litellm_main_logger.setLevel(getattr(logging, litellm_log_level.upper(), logging.INFO))

        litellm_proxy_logger = logging.getLogger("LiteLLM Proxy")
        litellm_proxy_logger.setLevel(getattr(logging, litellm_proxy_log_level.upper(), logging.INFO))

        litellm_router_logger = logging.getLogger("LiteLLM Router")
        litellm_router_logger.setLevel(getattr(logging, litellm_router_log_level.upper(), logging.INFO))

        litellm_format = (
            "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s"
        )
        litellm_formatter = logging.Formatter(litellm_format, datefmt="%Y-%m-%d %H:%M:%S")
        for lg in (litellm_main_logger, litellm_proxy_logger, litellm_router_logger):
            for h in lg.handlers:
                h.setFormatter(litellm_formatter)

        if health_check_enabled:
            benchmark = HealthBenchmark(
                test_prompt=health_check_prompt,
                max_latency_ms=health_check_max_latency_ms,
            )
            self._health_runner = HealthCheckRunner(benchmark=benchmark)
            self._start_router_poll_thread()

        logger.info(
            "RateLimitCallback initialized: cooldown=%.1fs, health_check=%s, probe_config=%s",
            default_cooldown_seconds,
            health_check_enabled,
            bool(models_to_check),
        )

    @classmethod
    def from_config(cls, config: RateLimitPluginConfig) -> "RateLimitCallback":
        return cls(
            default_cooldown_seconds=config.default_cooldown_seconds,
            models_to_check=config.health_check.models_to_check,
            health_check_enabled=config.health_check.enabled,
            health_check_interval_seconds=config.health_check.interval_seconds,
            health_check_prompt=config.health_check.test_prompt,
            health_check_max_latency_ms=config.health_check.max_latency_ms,
            log_level=config.logging.log_level,
            litellm_log_level=config.logging.litellm_log_level,
            litellm_proxy_log_level=config.logging.litellm_proxy_log_level,
            litellm_router_log_level=config.logging.litellm_router_log_level,
        )

    def _start_router_poll_thread(self) -> None:
        def poll_for_router():
            for _ in range(60):
                if self._health_checks_started:
                    return
                try:
                    from litellm.proxy.proxy_server import llm_router

                    if llm_router is not None:
                        logger.info("Router detected via poll thread, starting health checks")
                        self._router = llm_router
                        self._alias_state.set_router(llm_router)
                        self._build_model_mappings()
                        self._inject_alias_fallbacks()

                        if self._probe_config and hasattr(llm_router, "model_list"):
                            self._probe_config.build_from_router(llm_router.model_list)
                            logger.info("Built probe config from router model_list")

                        model_names = self._get_model_names_from_router()
                        if model_names:
                            self._health_checks_started = True
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                self._health_check_loop = loop
                                future = loop.create_future()
                                loop.run_until_complete(
                                    self._run_initial_checks_and_start_periodic(model_names, future)
                                )
                                # Keep the loop running to support periodic health checks
                                loop.run_forever()
                            except Exception as e:
                                logger.error("Failed to run startup health checks: %s", e)
                                if loop.is_running():
                                    loop.stop()
                                loop.close()
                        return
                except ImportError:
                    pass
                time.sleep(0.5)
            logger.warning("Router poll thread timed out waiting for router")

        self._startup_thread = threading.Thread(target=poll_for_router, daemon=True)
        self._startup_thread.start()
        logger.info("Started router poll thread for startup health checks")

    def set_router(self, router: "LiteLLMRouter") -> None:
        self._router = router
        self._alias_state.set_router(router)
        self._build_model_mappings()
        self._inject_alias_fallbacks()
        logger.info("Router reference set")

        if self._probe_config and hasattr(router, "model_list"):
            self._probe_config.build_from_router(router.model_list)
            logger.info("Built probe config from router model_list")

        if self._health_check_enabled and self._health_runner and not self._health_checks_started:
            model_names = self._get_model_names_from_router()
            if model_names:
                self._health_checks_started = True
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._run_initial_checks_and_start_periodic(model_names, None))
                    logger.info("Scheduled startup health checks for %d models", len(model_names))
                except RuntimeError:
                    self._startup_models = model_names
                    logger.info("No event loop, deferring startup health checks to first request")

    async def _run_initial_checks_and_start_periodic(
        self,
        model_names: list[str],
        initial_done_future: asyncio.Future | None = None,
    ) -> None:
        """Run initial health checks and start periodic checks.

        When initial_done_future is provided, it will be set when initial checks complete,
        but the periodic checks will continue running indefinitely (until stopped).
        """
        if self._health_runner is None:
            logger.warning("Health runner not initialized, skipping startup health checks")
            if initial_done_future and not initial_done_future.done():
                initial_done_future.set_result(None)
            return
        if not model_names:
            logger.warning("No models to health check at startup")
            if initial_done_future and not initial_done_future.done():
                initial_done_future.set_result(None)
            return

        models_to_check = (
            self._probe_config.get_models_to_health_check(model_names) if self._probe_config else model_names
        )
        logger.info(
            "Running startup health checks for %d models (reduced from %d via probe config)",
            len(models_to_check),
            len(model_names),
        )
        client = self._get_health_check_client()
        await self._health_runner.run_initial_checks_and_start_periodic(
            models=models_to_check,
            interval_seconds=self._health_check_interval,
            health_manager=self._health_state,
            client=client,
            cooldown_seconds=self.default_cooldown_seconds,
        )
        logger.info("Completed startup health checks for %d models", len(models_to_check))

        # Signal that initial checks are done (but periodic checks continue)
        if initial_done_future and not initial_done_future.done():
            initial_done_future.set_result(None)

        # Block forever to keep the periodic health checks running
        if self._health_check_loop and self._health_check_loop.is_running():
            await asyncio.Event().wait()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> dict:
        original_model = data.get("model", "")

        self._ensure_router()

        if self._startup_models:
            models = self._startup_models
            self._startup_models = None
            logger.info("Triggering startup health checks for %d models", len(models))
            await self._run_initial_checks_and_start_periodic(models, None)

        rate_limited = await self._health_state.is_rate_limited(original_model)
        resolved_target = original_model

        if not rate_limited:
            rate_limited = await self._alias_state.is_rate_limited(original_model)

        if rate_limited:
            resolved_target = self._alias_state._resolve_to_target(original_model)
            logger.info(
                "Model %s is rate-limited (resolved to %s), adding to cooldown cache",
                original_model,
                resolved_target,
            )
            await self._sync_health_state_to_cooldown(resolved_target, data)

        return data

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        traceback_str: str | None = None,
    ) -> None:
        await self._handle_deployment_failure(
            exception=original_exception,
            model=request_data.get("model", "unknown"),
            request_data=request_data,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        exception = kwargs.get("exception")
        if exception is None:
            exception = kwargs.get("original_exception")

        model = kwargs.get("model", "unknown")

        litellm_params = kwargs.get("litellm_params") or {}
        litellm_model = litellm_params.get("model", "") if isinstance(litellm_params, dict) else ""

        if litellm_model and model != litellm_model:
            resolved = self._resolve_model_from_litellm_model(litellm_model)
            if resolved:
                logger.debug(
                    "log_failure_event: resolved model %s -> %s from litellm_model=%s",
                    model,
                    resolved,
                    litellm_model,
                )
                model = resolved

        logger.debug(
            "log_failure_event: model=%s, litellm_model=%s, has_exception=%s",
            model,
            litellm_model,
            exception is not None,
        )

        if exception is None:
            return

        await self._handle_deployment_failure(
            exception=exception,
            model=model,
            request_data=kwargs,
        )

    async def log_failure_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict,
        original_exception: Exception,
    ) -> None:
        """Called by LiteLLM Router for EACH failed fallback model during fallback chains.

        This hook bypasses the ``has_logged_async_failure`` dedup that blocks
        ``async_log_failure_event`` for fallback model failures.  It fires once
        per failed fallback model, giving us the model name in ``kwargs["model"]``.
        """
        model = kwargs.get("model", "")
        if not model:
            return

        if await self._health_state.is_rate_limited(model):
            return
        if await self._alias_state.is_rate_limited(model):
            return

        logger.info(
            "Fallback model %s failed (original request: %s), marking unhealthy",
            model,
            original_model_group,
        )

        cooldown_seconds = self._get_cooldown_for_model(model)
        await self._alias_state.mark_rate_limited(model, cooldown_seconds)

        router = self._ensure_router()
        if router is not None:
            resolved_model = self._alias_state._resolve_to_target(model)
            await self._update_cooldown(resolved_model, cooldown_seconds, kwargs)

    async def log_success_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict,
        original_exception: Exception,
    ) -> None:
        """Called when a fallback chain succeeds.

        Infers which intermediate models in the chain must have failed and marks
        them unhealthy.  This is a safety-net that catches models missed by
        ``log_failure_fallback_event``.
        """
        successful_model = kwargs.get("model", "")
        if not successful_model or successful_model == original_model_group:
            return

        failed_models = self._infer_failed_fallback_models(
            original_model_group,
            successful_model,
        )

        for model in failed_models:
            if await self._health_state.is_rate_limited(model):
                continue
            if await self._alias_state.is_rate_limited(model):
                continue

            logger.info(
                "Inferred fallback model %s failed (original: %s, succeeded: %s), marking unhealthy",
                model,
                original_model_group,
                successful_model,
            )

            cooldown_seconds = self._get_cooldown_for_model(model)
            await self._alias_state.mark_rate_limited(model, cooldown_seconds)

            router = self._ensure_router()
            if router is not None:
                resolved_model = self._alias_state._resolve_to_target(model)
                await self._update_cooldown(resolved_model, cooldown_seconds)

    def _infer_failed_fallback_models(
        self,
        original_model_group: str,
        successful_model: str,
    ) -> list[str]:
        """Return models that must have failed between *original_model_group* and
        *successful_model* in the fallback chain."""
        if not self._router:
            return []
        fallbacks = getattr(self._router, "fallbacks", None)
        if not fallbacks:
            return []
        for item in fallbacks:
            if isinstance(item, dict) and original_model_group in item:
                chain = item[original_model_group]
                if not isinstance(chain, list):
                    continue
                failed: list[str] = []
                for m in chain:
                    if m == successful_model:
                        break
                    failed.append(m)
                return failed
        return []

    async def async_post_call_success_hook(
        self,
        data: dict,
        response: object,
        user_api_key_dict: UserAPIKeyAuth | None = None,
    ) -> None:
        requested_model = data.get("model", "unknown")

        actual_model, deployment_id = self._resolve_actual_model_with_deployment(
            data=data,
            response=response,
            requested_model=requested_model,
        )

        if actual_model != requested_model:
            logger.info(
                "Successfully called model %s (requested: %s, deployment: %s)",
                actual_model,
                requested_model,
                deployment_id or "unknown",
            )
        else:
            logger.info(
                "Successfully called model %s (deployment: %s)",
                actual_model,
                deployment_id or "unknown",
            )

    def _resolve_actual_model_with_deployment(
        self,
        data: dict,
        response: object,
        requested_model: str,
    ) -> tuple[str, str | None]:
        """Resolve the actual model name used and its deployment ID, handling fallback scenarios.

        Resolution order:
        1. litellm_params.model in data (direct API call)
        2. response.model (fallback — contains provider-prefixed actual model)
        3. deployment_id lookup via litellm_params.model_info.id
        4. requested_model as-is (no resolution possible)

        Returns:
            tuple of (actual_model_name, deployment_id)
        """
        litellm_params = data.get("litellm_params") or {}

        # Step 1: Try litellm_params.model (the litellm_model string, e.g. "minimax/MiniMax-M2.5")
        litellm_model = litellm_params.get("model", "") if isinstance(litellm_params, dict) else ""
        if litellm_model:
            resolved = self._resolve_model_from_litellm_model(litellm_model)
            if resolved:
                deployment_id = self._get_deployment_id_for_model(resolved)
                return resolved, deployment_id

        # Step 2: Try response.model — could be a litellm_model OR a model_name
        if hasattr(response, "model") and response.model:
            resolved = self._resolve_model_from_litellm_model(response.model)
            if resolved:
                deployment_id = self._get_deployment_id_for_model(resolved)
                return resolved, deployment_id

            # Try response.model as a model_name directly
            deployment_id = self._get_deployment_id_for_model(response.model)
            if deployment_id:
                return response.model, deployment_id

            # Only return response.model as-is if it differs from requested_model
            # (avoids returning the alias as the actual model)
            if response.model != requested_model:
                return response.model, None

        # Step 3: Try litellm_params.model_info.id to look up model_name and deployment
        model_info = litellm_params.get("model_info") if isinstance(litellm_params, dict) else None
        deployment_id = model_info.get("id") if isinstance(model_info, dict) else None
        if deployment_id:
            resolved_model_name = self._get_model_name_for_deployment(deployment_id)
            if resolved_model_name:
                return resolved_model_name, deployment_id

        # Step 4: No resolution possible — return requested model as-is
        return requested_model, None

    def _get_deployment_id_for_model(self, model_name: str) -> str | None:
        """Look up deployment ID for a given model name in the router's model_list."""
        if not hasattr(self._router, "model_list"):
            return None
        for deployment in self._router.model_list:
            model_name_in_deployment = (
                deployment.get("model_name")
                if isinstance(deployment, dict)
                else getattr(deployment, "model_name", None)
            )
            if model_name_in_deployment != model_name:
                continue
            model_info = (
                deployment.get("model_info", {})
                if isinstance(deployment, dict)
                else getattr(deployment, "model_info", {})
            )
            dep_id = model_info.get("id") if isinstance(model_info, dict) else getattr(model_info, "id", None)
            if dep_id:
                return dep_id
        return None

    def _get_model_name_for_deployment(self, deployment_id: str) -> str | None:
        """Look up model_name from a deployment ID in the router's model_list."""
        if not hasattr(self._router, "model_list"):
            return None
        for deployment in self._router.model_list:
            model_info = (
                deployment.get("model_info", {})
                if isinstance(deployment, dict)
                else getattr(deployment, "model_info", {})
            )
            dep_id = model_info.get("id") if isinstance(model_info, dict) else getattr(model_info, "id", None)
            if dep_id == deployment_id:
                return (
                    deployment.get("model_name")
                    if isinstance(deployment, dict)
                    else getattr(deployment, "model_name", None)
                )
        return None

    def _ensure_router(self) -> "LiteLLMRouter | None":
        if self._router is not None:
            return self._router
        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is not None:
                self._router = llm_router
                self._alias_state.set_router(llm_router)
                self._build_model_mappings()
                self._inject_alias_fallbacks()
                logger.info("Router reference obtained from proxy global")

                if self._probe_config and hasattr(llm_router, "model_list"):
                    self._probe_config.build_from_router(llm_router.model_list)
                    logger.info("Built probe config from router model_list")

                if self._health_check_enabled and self._health_runner and not self._health_checks_started:
                    model_names = self._get_model_names_from_router()
                    if model_names:
                        self._health_checks_started = True
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(self._run_initial_checks_and_start_periodic(model_names, None))
                            logger.info(
                                "Auto-started health checks for %d models via _ensure_router",
                                len(model_names),
                            )
                        except RuntimeError:
                            self._startup_models = model_names
                            logger.info("No event loop, deferring health checks to first request")
        except ImportError:
            pass
        return self._router

    async def _handle_deployment_failure(
        self, exception: Exception, model: str, request_data: dict | None = None
    ) -> None:
        logger.debug(
            "Handling deployment failure for model %s: exception_type=%s, has_status_code=%s, str=%.200s",
            model,
            type(exception).__name__,
            hasattr(exception, "status_code"),
            str(exception),
        )

        if isinstance(exception, ProxyException):
            logger.debug(
                "ProxyException (auth/config error), skipping cooldown for %s: %s",
                model,
                exception,
            )
            return

        detected = detect_api_error(exception)
        if detected is not None:
            is_err, status_code, _ = detected
            if status_code in self._SKIP_STATUS_CODES:
                logger.debug("Auth/permission error (%s), skipping", status_code)
                return
        else:
            logger.debug("No error detected from exception, skipping")
            return

        status_code = None
        if detected:
            _, status_code, _ = detected

        if is_rate_limit_error(exception):
            cooldown_seconds = extract_rate_limit_reset_seconds(
                exception,
                default=self._get_cooldown_for_model(model, request_data),
            )
            from litellm_rate_limit.parser import extract_rate_limit_reset_dt

            reset_at = extract_rate_limit_reset_dt(exception)
        else:
            cooldown_seconds = self._get_cooldown_for_model(model)
            reset_at = None

        logger.info(
            "Marking model %s unhealthy for %.1f seconds (status=%s, exception=%s)",
            model,
            cooldown_seconds,
            status_code,
            type(exception).__name__,
        )

        await self._alias_state.mark_rate_limited(model, cooldown_seconds, reset_at=reset_at)

        router = self._ensure_router()
        if router is not None:
            resolved_model = self._alias_state._resolve_to_target(model)
            await self._update_cooldown(resolved_model, cooldown_seconds, request_data)

    def _get_cooldown_for_model(self, model: str, request_data: dict | None = None) -> float:
        return self.default_cooldown_seconds

    async def _update_cooldown(
        self, model: str, cooldown_seconds: float, request_data: dict | None = None
    ) -> None:
        if not hasattr(self._router, "cooldown_cache"):
            logger.debug("Router has no cooldown_cache attribute")
            return

        async with self._cooldown_cache_lock:
            deployment_id = None
            if request_data is not None:
                litellm_params = request_data.get("litellm_params") or {}
                model_info = litellm_params.get("model_info") if isinstance(litellm_params, dict) else None
                deployment_id = model_info.get("id") if isinstance(model_info, dict) else None

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

    async def _sync_health_state_to_cooldown(self, model: str, request_data: dict | None = None) -> None:
        if self._router is None or not hasattr(self._router, "cooldown_cache"):
            return

        remaining = self._health_state.get_remaining_cooldown(model)
        if remaining is None:
            remaining = self._alias_state.get_remaining_cooldown(model)
        cooldown_seconds = (
            remaining if remaining is not None else self._get_cooldown_for_model(model, request_data)
        )

        deployment_ids = self._get_deployment_ids_for_model(model)
        if deployment_ids:
            for dep_id in deployment_ids:
                if self._is_deployment_in_cooldown(dep_id):
                    logger.info(
                        "Deployment %s (model %s) already in cooldown, skipping sync",
                        dep_id,
                        model,
                    )
                    continue
                self._router.cooldown_cache.add_deployment_to_cooldown(
                    model_id=dep_id,
                    original_exception=Exception("Model rate-limited by health check"),
                    exception_status=429,
                    cooldown_time=cooldown_seconds,
                )
                logger.debug(
                    "Synced health state to cooldown for deployment %s (model %s): %.1fs",
                    dep_id,
                    model,
                    cooldown_seconds,
                )
        else:
            if self._is_deployment_in_cooldown(model):
                return
            self._router.cooldown_cache.add_deployment_to_cooldown(
                model_id=model,
                original_exception=Exception("Model rate-limited by health check"),
                exception_status=429,
                cooldown_time=cooldown_seconds,
            )
            logger.info(
                "Synced health state to cooldown for model %s (fallback): %.1fs",
                model,
                cooldown_seconds,
            )

    def _is_deployment_in_cooldown(self, deployment_id: str) -> bool:
        if not hasattr(self._router, "cooldown_cache"):
            return False
        get_active = getattr(self._router.cooldown_cache, "get_active_cooldowns", None)
        if get_active is None:
            return False
        active = get_active(
            model_ids=[deployment_id],
            parent_otel_span=None,
        )
        return bool(active)

    def _resolve_model_from_litellm_model(self, litellm_model: str) -> str | None:
        if not litellm_model or not self._model_name_to_litellm_model:
            return None
        for model_name, ll_model in self._model_name_to_litellm_model.items():
            if ll_model == litellm_model:
                return model_name
        return None

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
        if self._router is None:
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

    def _build_model_mappings(self) -> None:
        if self._router is None:
            return
        model_list = getattr(self._router, "model_list", None)
        if model_list is None:
            return
        try:
            iter(model_list)
        except TypeError:
            return
        self._model_name_to_litellm_model.clear()
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

    def _inject_alias_fallbacks(self) -> None:
        if self._router is None:
            return

        alias_map = getattr(self._router, "model_group_alias", None)
        fallbacks = getattr(self._router, "fallbacks", None)
        if not alias_map or not fallbacks:
            return

        try:
            iter(fallbacks)
        except TypeError:
            return

        existing_keys = set()
        for item in fallbacks:
            if isinstance(item, dict):
                existing_keys.add(list(item.keys())[0])

        injected = []
        for alias_name, target in alias_map.items():
            if isinstance(target, dict):
                target = target.get("model", alias_name)
            target_str = str(target)

            if alias_name in existing_keys:
                continue

            for item in fallbacks:
                if isinstance(item, dict) and target_str in item:
                    injected.append({alias_name: item[target_str]})
                    logger.info(
                        "Injected alias fallback: %s -> %s (mirrors target %s)",
                        alias_name,
                        item[target_str],
                        target_str,
                    )
                    break

        if injected:
            fallbacks.extend(injected)

    def _get_models_to_health_check(self, all_models: list[str]) -> list[str]:
        if not all_models:
            return []

        if not self._probe_config or not self._probe_config.models_to_check:
            return all_models

        return self._probe_config.get_models_to_health_check(all_models)

    def _get_health_check_client(self) -> Callable:
        router = self._router
        model_mapping = self._model_name_to_litellm_model

        async def client(model_id: str, prompt: str):
            if router is None:
                raise RuntimeError("Router not set, cannot run health check")
            litellm_model = model_mapping.get(model_id, model_id)
            return await router.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
            )

        return client

    async def start_health_checks(self, models: list[str]) -> None:
        if not self._health_runner:
            logger.warning("Health checker not enabled")
            return

        client = self._get_health_check_client()
        await self._health_runner.start_periodic_checks(
            name="default",
            models=models,
            interval_seconds=self._health_check_interval,
            health_manager=self._health_state,
            client=client,
            cooldown_seconds=self.default_cooldown_seconds,
        )

    async def stop_health_checks(self) -> None:
        if self._health_runner:
            await self._health_runner.stop_all()
            logger.info("Stopped health checks")

    @property
    def health_state(self) -> HealthStateManager:
        return self._health_state
