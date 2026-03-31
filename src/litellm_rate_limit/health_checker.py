"""Background health checker for proactive model benchmarking."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from litellm_rate_limit.parser import detect_api_error

if TYPE_CHECKING:
    from litellm_rate_limit.health_state import HealthStateManager

logger = logging.getLogger(__name__)

_original_litellm_logger = None
_original_litellm_router_logger = None


def suppress_litellm_error_logs() -> None:
    """Suppress LiteLLM's verbose error logging during health checks."""
    global _original_litellm_logger, _original_litellm_router_logger

    for logger_name in ["litellm", "litellm.router"]:
        lg = logging.getLogger(logger_name)
        if not hasattr(lg, "_health_check_original_level"):
            lg._health_check_original_level = lg.level
            lg.setLevel(logging.CRITICAL)


def restore_litellm_error_logs() -> None:
    """Restore LiteLLM's error logging after health checks."""
    for logger_name in ["litellm", "litellm.router"]:
        lg = logging.getLogger(logger_name)
        if hasattr(lg, "_health_check_original_level"):
            lg.setLevel(lg._health_check_original_level)
            delattr(lg, "_health_check_original_level")


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    model_id: str
    status: HealthStatus
    latency_ms: float | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.monotonic)
    response_valid: bool = True


def _is_error_response(result: object) -> tuple[bool, str]:
    detected = detect_api_error(result)
    if detected is not None:
        return (True, detected[2])
    return (False, "")


@dataclass
class HealthBenchmark:
    """Runs health checks on models to measure latency and availability."""

    test_prompt: str = "Say 'ok'"
    timeout_seconds: float = 30.0
    max_latency_ms: float = 30000.0

    _check_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(5))

    async def run_health_check(
        self,
        model_id: str,
        client: Callable | None = None,
    ) -> HealthCheckResult:
        async with self._check_semaphore:
            try:
                start_time = time.monotonic()

                if client is None:
                    await asyncio.sleep(0.01)
                    latency_ms = (time.monotonic() - start_time) * 1000
                    return HealthCheckResult(
                        model_id=model_id,
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency_ms,
                    )

                result = await asyncio.wait_for(
                    client(model_id, self.test_prompt),
                    timeout=self.timeout_seconds,
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                if latency_ms > self.max_latency_ms:
                    return HealthCheckResult(
                        model_id=model_id,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=f"Latency {latency_ms:.0f}ms exceeds max {self.max_latency_ms:.0f}ms",
                        response_valid=True,
                    )

                detected = detect_api_error(result)
                if detected is not None:
                    _, _, error_msg = detected
                    return HealthCheckResult(
                        model_id=model_id,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=error_msg,
                        response_valid=False,
                    )

                return HealthCheckResult(
                    model_id=model_id,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    response_valid=result is not None,
                )

            except asyncio.TimeoutError:
                return HealthCheckResult(
                    model_id=model_id,
                    status=HealthStatus.UNHEALTHY,
                    error=f"Health check timed out after {self.timeout_seconds}s",
                )
            except Exception as e:
                return HealthCheckResult(
                    model_id=model_id,
                    status=HealthStatus.UNHEALTHY,
                    error=str(e),
                )

    async def run_periodic_checks(
        self,
        models: list[str],
        interval_seconds: int = 60,
        health_manager: Optional["HealthStateManager"] = None,
        client: Callable | None = None,
        stop_event: asyncio.Event | None = None,
        cooldown_seconds: float = 3600.0,
    ) -> None:
        if stop_event is None:
            stop_event = asyncio.Event()

        while not stop_event.is_set():
            logger.info("Running health checks for %d models", len(models))

            suppress_litellm_error_logs()
            try:
                results = await asyncio.gather(
                    *[self.run_health_check(model, client) for model in models],
                    return_exceptions=True,
                )
            except Exception as e:
                restore_litellm_error_logs()
                logger.error("Health check gather failed: %s", e)
                raise
            restore_litellm_error_logs()

            for result in results:
                if isinstance(result, Exception):
                    logger.error("Health check failed with exception: %s", result)
                    continue

                if health_manager is not None:
                    if result.status == HealthStatus.HEALTHY:
                        await health_manager.record_success(result.model_id)
                        await health_manager.clear_rate_limit(result.model_id)
                    else:
                        await health_manager.mark_rate_limited(
                            result.model_id,
                            cooldown_seconds,
                        )

                log_level = logging.INFO if result.status == HealthStatus.HEALTHY else logging.WARNING
                logger.log(
                    log_level,
                    "Health check for %s: %s (latency: %.0fms)",
                    result.model_id,
                    result.status.value,
                    result.latency_ms or 0,
                )

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)


@dataclass
class HealthCheckRunner:
    """Manages background health check tasks."""

    benchmark: HealthBenchmark = field(default_factory=HealthBenchmark)
    _running_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    _stop_events: dict[str, asyncio.Event] = field(default_factory=dict)

    async def start_periodic_checks(
        self,
        name: str,
        models: list[str],
        interval_seconds: int = 60,
        health_manager: Optional["HealthStateManager"] = None,
        client: Callable | None = None,
        cooldown_seconds: float = 3600.0,
    ) -> None:
        if name in self._running_tasks:
            logger.warning("Health check task '%s' already running", name)
            return

        stop_event = asyncio.Event()
        self._stop_events[name] = stop_event

        task = asyncio.create_task(
            self.benchmark.run_periodic_checks(
                models=models,
                interval_seconds=interval_seconds,
                health_manager=health_manager,
                client=client,
                stop_event=stop_event,
                cooldown_seconds=cooldown_seconds,
            ),
            name=f"health-check-{name}",
        )

        self._running_tasks[name] = task
        logger.info("Started health check task '%s' for %d models", name, len(models))

    async def run_initial_checks_and_start_periodic(
        self,
        models: list[str],
        interval_seconds: int = 60,
        health_manager: Optional["HealthStateManager"] = None,
        client: Callable | None = None,
        cooldown_seconds: float = 3600.0,
    ) -> None:
        logger.info("Running initial health checks for %d models", len(models))
        suppress_litellm_error_logs()
        try:
            results = await asyncio.gather(
                *[self.benchmark.run_health_check(model, client) for model in models],
                return_exceptions=True,
            )
        except Exception as e:
            restore_litellm_error_logs()
            logger.error("Initial health check gather failed: %s", e)
            raise
        restore_litellm_error_logs()

        healthy_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error("Initial health check failed with exception: %s", result)
                continue

            if health_manager is not None:
                if result.status == HealthStatus.HEALTHY:
                    await health_manager.record_success(result.model_id)
                    await health_manager.clear_rate_limit(result.model_id)
                    healthy_count += 1
                else:
                    await health_manager.mark_rate_limited(result.model_id, cooldown_seconds)

            log_level = logging.INFO if result.status == HealthStatus.HEALTHY else logging.WARNING
            logger.log(
                log_level,
                "Health check for %s: %s (latency: %.0fms)",
                result.model_id,
                result.status.value,
                result.latency_ms or 0,
            )

        logger.info(
            "Initial health checks complete: %d/%d models healthy",
            healthy_count,
            len(models),
        )

        await self.start_periodic_checks(
            name="startup",
            models=models,
            interval_seconds=interval_seconds,
            health_manager=health_manager,
            client=client,
            cooldown_seconds=cooldown_seconds,
        )

    async def stop_periodic_checks(self, name: str) -> bool:
        if name not in self._running_tasks:
            return False

        self._stop_events[name].set()
        task = self._running_tasks[name]
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        del self._running_tasks[name]
        del self._stop_events[name]
        logger.info("Stopped health check task '%s'", name)
        return True

    async def stop_all(self) -> None:
        for name in list(self._running_tasks.keys()):
            await self.stop_periodic_checks(name)

    def is_running(self, name: str) -> bool:
        return name in self._running_tasks and not self._running_tasks[name].done()

    def get_running_tasks(self) -> list[str]:
        return [name for name, task in self._running_tasks.items() if not task.done()]
