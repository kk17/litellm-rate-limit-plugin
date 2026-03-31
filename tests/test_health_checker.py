"""Unit tests for health checker."""

import asyncio
import contextlib
from unittest.mock import Mock

import pytest

from litellm_rate_limit.health_checker import (
    HealthBenchmark,
    HealthCheckResult,
    HealthCheckRunner,
    HealthStatus,
)


class TestHealthStatus:
    def test_health_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    def test_health_check_result_creation(self):
        result = HealthCheckResult(
            model_id="claude-3-sonnet",
            status=HealthStatus.HEALTHY,
            latency_ms=150.5,
        )
        assert result.model_id == "claude-3-sonnet"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 150.5
        assert result.response_valid is True

    def test_health_check_result_with_error(self):
        result = HealthCheckResult(
            model_id="claude-3-sonnet",
            status=HealthStatus.UNHEALTHY,
            error="Connection timeout",
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error == "Connection timeout"


class TestHealthBenchmark:
    def test_health_benchmark_defaults(self):
        benchmark = HealthBenchmark()
        assert benchmark.test_prompt == "Say 'ok'"
        assert benchmark.timeout_seconds == 30.0
        assert benchmark.max_latency_ms == 30000.0

    @pytest.mark.asyncio
    async def test_run_health_check_success(self):
        benchmark = HealthBenchmark()

        result = await benchmark.run_health_check("claude-3-sonnet")

        assert result.status == HealthStatus.HEALTHY
        assert result.model_id == "claude-3-sonnet"
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_run_health_check_with_client(self):
        benchmark = HealthBenchmark()

        async def mock_client(model_id: str, prompt: str):
            return {"content": "ok"}

        result = await benchmark.run_health_check("claude-3-sonnet", client=mock_client)

        assert result.status == HealthStatus.HEALTHY
        assert result.response_valid is True

    @pytest.mark.asyncio
    async def test_run_health_check_failure(self):
        benchmark = HealthBenchmark()

        async def failing_client(model_id: str, prompt: str):
            raise RuntimeError("API error")

        result = await benchmark.run_health_check("claude-3-sonnet", client=failing_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_run_health_check_timeout(self):
        benchmark = HealthBenchmark(timeout_seconds=0.1)

        async def slow_client(model_id: str, prompt: str):
            await asyncio.sleep(1.0)
            return {"content": "ok"}

        result = await benchmark.run_health_check("claude-3-sonnet", client=slow_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_health_check_high_latency(self):
        benchmark = HealthBenchmark(max_latency_ms=0.001)

        async def mock_client(model_id: str, prompt: str):
            await asyncio.sleep(0.01)
            return {"content": "ok"}

        result = await benchmark.run_health_check("claude-3-sonnet", client=mock_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert "exceeds max" in result.error

    @pytest.mark.asyncio
    async def test_run_health_check_error_response_dict(self):
        benchmark = HealthBenchmark()

        async def error_client(model_id: str, prompt: str):
            return {"error": {"message": "Usage limit reached", "code": "quota_exceeded"}}

        result = await benchmark.run_health_check("grok-code-fast-1", client=error_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert result.response_valid is False
        assert "Usage limit reached" in result.error

    @pytest.mark.asyncio
    async def test_run_health_check_error_response_object(self):
        benchmark = HealthBenchmark()

        class MockResponse:
            def __init__(self):
                self.error = {"message": "No quota", "type": "insufficient_quota"}

        async def error_client(model_id: str, prompt: str):
            return MockResponse()

        result = await benchmark.run_health_check("grok-code-fast-1", client=error_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert result.response_valid is False
        assert "No quota" in result.error

    @pytest.mark.asyncio
    async def test_run_health_check_error_status_code(self):
        benchmark = HealthBenchmark()

        class MockResponse:
            def __init__(self):
                self.status_code = 402

        async def error_client(model_id: str, prompt: str):
            return MockResponse()

        result = await benchmark.run_health_check("grok-code-fast-1", client=error_client)

        assert result.status == HealthStatus.UNHEALTHY
        assert result.response_valid is False
        assert "402" in result.error

    @pytest.mark.asyncio
    async def test_run_health_check_success_response(self):
        benchmark = HealthBenchmark()

        async def success_client(model_id: str, prompt: str):
            return {"choices": [{"message": {"content": "ok"}}]}

        result = await benchmark.run_health_check("gpt-4o", client=success_client)

        assert result.status == HealthStatus.HEALTHY
        assert result.response_valid is True


class TestIsErrorResponse:
    def test_clean_dict_response(self):
        from litellm_rate_limit.health_checker import _is_error_response

        is_err, msg = _is_error_response({"choices": [{"message": {"content": "ok"}}]})
        assert is_err is False

    def test_error_dict_with_message(self):
        from litellm_rate_limit.health_checker import _is_error_response

        is_err, msg = _is_error_response({"error": {"message": "Rate limit", "type": "rate_limit_error"}})
        assert is_err is True
        assert "Rate limit" in msg

    def test_error_dict_string(self):
        from litellm_rate_limit.health_checker import _is_error_response

        is_err, msg = _is_error_response({"error": "something went wrong"})
        assert is_err is True
        assert "something went wrong" in msg

    def test_object_with_error_attr(self):
        from litellm_rate_limit.health_checker import _is_error_response

        class Obj:
            error = {"message": "quota exceeded"}

        is_err, msg = _is_error_response(Obj())
        assert is_err is True
        assert "quota exceeded" in msg

    def test_object_with_error_string(self):
        from litellm_rate_limit.health_checker import _is_error_response

        class Obj:
            error = "Not found"

        is_err, msg = _is_error_response(Obj())
        assert is_err is True
        assert "Not found" in msg

    def test_object_with_status_code_200(self):
        from litellm_rate_limit.health_checker import _is_error_response

        class Obj:
            status_code = 200

        is_err, msg = _is_error_response(Obj())
        assert is_err is False

    def test_object_with_status_code_402(self):
        from litellm_rate_limit.health_checker import _is_error_response

        class Obj:
            status_code = 402

        is_err, msg = _is_error_response(Obj())
        assert is_err is True
        assert "402" in msg

    def test_object_with_status_code_500(self):
        from litellm_rate_limit.health_checker import _is_error_response

        class Obj:
            status_code = 500

        is_err, msg = _is_error_response(Obj())
        assert is_err is True
        assert "500" in msg


class TestHealthCheckRunner:
    def test_health_check_runner_init(self):
        runner = HealthCheckRunner()
        assert runner._running_tasks == {}
        assert runner._stop_events == {}

    @pytest.mark.asyncio
    async def test_start_periodic_checks(self):
        runner = HealthCheckRunner()

        await runner.start_periodic_checks(
            name="test-check",
            models=["model-a", "model-b"],
            interval_seconds=1,
        )

        assert "test-check" in runner._running_tasks
        assert runner.is_running("test-check") is True

        await runner.stop_periodic_checks("test-check")

    @pytest.mark.asyncio
    async def test_stop_periodic_checks(self):
        runner = HealthCheckRunner()

        await runner.start_periodic_checks(
            name="test-check",
            models=["model-a"],
            interval_seconds=1,
        )

        result = await runner.stop_periodic_checks("test-check")

        assert result is True
        assert runner.is_running("test-check") is False

    @pytest.mark.asyncio
    async def test_stop_periodic_checks_not_found(self):
        runner = HealthCheckRunner()
        result = await runner.stop_periodic_checks("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_all(self):
        runner = HealthCheckRunner()

        await runner.start_periodic_checks("check-a", ["model-a"], interval_seconds=1)
        await runner.start_periodic_checks("check-b", ["model-b"], interval_seconds=1)

        await runner.stop_all()

        assert runner.is_running("check-a") is False
        assert runner.is_running("check-b") is False

    @pytest.mark.asyncio
    async def test_duplicate_start_warning(self):
        runner = HealthCheckRunner()

        await runner.start_periodic_checks("test-check", ["model-a"], interval_seconds=1)
        await runner.start_periodic_checks("test-check", ["model-b"], interval_seconds=1)

        assert len([n for n in runner._running_tasks if n == "test-check"]) == 1

        await runner.stop_all()

    def test_get_running_tasks(self):
        runner = HealthCheckRunner()

        runner._running_tasks["task-a"] = Mock(done=Mock(return_value=False))
        runner._running_tasks["task-b"] = Mock(done=Mock(return_value=True))

        running = runner.get_running_tasks()

        assert "task-a" in running
        assert "task-b" not in running


class TestPeriodicChecks:
    @pytest.mark.asyncio
    async def test_periodic_checks_iterations(self):
        benchmark = HealthBenchmark()
        stop_event = asyncio.Event()
        call_count = 0

        async def counting_client(model_id: str, prompt: str):
            nonlocal call_count
            call_count += 1
            return {"content": "ok"}

        task = asyncio.create_task(
            benchmark.run_periodic_checks(
                models=["model-a"],
                interval_seconds=0.1,
                client=counting_client,
                stop_event=stop_event,
            )
        )

        await asyncio.sleep(0.25)
        stop_event.set()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert call_count >= 2
