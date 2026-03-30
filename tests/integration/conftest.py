"""Pytest fixtures for integration tests."""

import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import httpx
import pytest

_port_counter = itertools.count(4001)


class OriginFormTransport(httpx.AsyncHTTPTransport):
    async def _handle_async_request(self, request):
        if request.url.is_absolute_url:
            from httpx._models import URL

            path = request.url.path
            if request.url.query:
                path = f"{path}?{request.url.query}"
            request = request.copy(url=URL(path))
        return await super()._handle_async_request(request)


class OriginFormSyncTransport(httpx.HTTPTransport):
    def _handle_request(self, request):
        if request.url.is_absolute_url:
            from httpx._models import URL

            path = request.url.path
            if request.url.query:
                path = f"{path}?{request.url.query}"
            request = request.copy(url=URL(path))
        return super()._handle_request(request)


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests with real LiteLLM proxy",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Mock API Server (session-scoped, shared by all tests)
# ---------------------------------------------------------------------------


class MockAPIHandler(BaseHTTPRequestHandler):
    rate_limited_models: set = set()
    call_counts: dict = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        data = json.loads(body) if body else {}
        model = data.get("model", "unknown")

        self.__class__.call_counts[model] = self.__class__.call_counts.get(model, 0) + 1

        if model in self.__class__.rate_limited_models:
            response_body = json.dumps(
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                }
            ).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(response_body))
            self.send_header("retry-after", "1")
            self.send_header("x-ratelimit-reset", "1")
            self.end_headers()
            self.wfile.write(response_body)
            return

        response_body = json.dumps(
            {
                "id": f"chatcmpl-{model}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Response from {model}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


@pytest.fixture(scope="session")
def mock_api_server():
    server = ThreadingHTTPServer(("localhost", 8765), MockAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield MockAPIHandler
    server.shutdown()


@pytest.fixture
def mock_api_control(mock_api_server):
    mock_api_server.rate_limited_models.clear()
    mock_api_server.call_counts.clear()
    return mock_api_server


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """
general_settings:
  master_key: test-master-key

model_list:
  - model_name: test-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: sk-test-key
      api_base: http://localhost:8765
      num_retries: 0
      request_timeout: 10
      allowed_fails: 100
      health_check: false

litellm_settings:
  callbacks: ["callback_for_test.rate_limit_callback"]
  num_retries: 0
  enable_pre_call_checks: false

rate_limit_plugin:
  default_cooldown_seconds: 1
  probe_models_by_provider:
    openai: ["gpt-4o-mini"]
"""


def _write_config(config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    callback_src = Path(__file__).parent / "callback_for_test.py"
    shutil.copy(callback_src, config_dir / "callback_for_test.py")
    config_path = config_dir / "config.yaml"
    config_path.write_text(_CONFIG_TEMPLATE)
    return config_path


def _find_litellm_bin() -> Path:
    venv_path = Path(__file__).parent.parent.parent / ".venv"
    litellm_bin = venv_path / "bin" / "litellm"
    if not litellm_bin.exists():
        litellm_bin = venv_path / "Scripts" / "litellm.exe"
    if not litellm_bin.exists():
        litellm_bin = Path(sys.executable).parent / "litellm"
    return litellm_bin


def _start_proxy(config_path: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent / "src")

    proc = subprocess.Popen(
        [str(_find_litellm_bin()), "--config", str(config_path), "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    max_wait = 60
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            with httpx.Client(transport=OriginFormSyncTransport(), timeout=1.0) as client:
                client.get(f"http://localhost:{port}/")
            return proc
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)

    proc.kill()
    stdout, stderr = proc.communicate(timeout=5)
    raise RuntimeError(f"LiteLLM proxy failed to start on port {port}.\nstdout: {stdout}\nstderr: {stderr}")


def _stop_proxy(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Proxy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integration_config_path(tmp_path_factory) -> Path:
    config_dir = tmp_path_factory.mktemp("litellm_config_shared")
    return _write_config(config_dir)


@pytest.fixture(scope="session")
def litellm_proxy(integration_config_path, mock_api_server):
    proc = _start_proxy(integration_config_path, 4001)
    yield {
        "base_url": "http://localhost:4001",
        "process": proc,
    }
    _stop_proxy(proc)


@pytest.fixture
def proxy_client(litellm_proxy):
    with httpx.Client(timeout=30.0, transport=OriginFormSyncTransport()) as client:
        yield client


@pytest.fixture
def per_test_proxy(tmp_path, mock_api_server):
    """Function-scoped: fresh LiteLLM process per test on a unique port.

    Use this for tests that put deployments into cooldown, so state never
    leaks between tests.
    """
    port = next(_port_counter)
    config_path = _write_config(tmp_path / "litellm_config")
    proc = _start_proxy(config_path, port)
    yield {
        "base_url": f"http://localhost:{port}",
        "process": proc,
    }
    _stop_proxy(proc)
