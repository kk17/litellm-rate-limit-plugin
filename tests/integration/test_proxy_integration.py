"""Integration tests with real LiteLLM proxy."""

import json
import subprocess

import pytest


def curl_get(url: str, headers: dict = None, timeout: int = 10) -> tuple[int, str]:
    cmd = ["curl", "-s", "-X", "GET", url, "-w", "\\n%{http_code}", "--noproxy", "*"]
    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = r.stdout.strip()
    status_code = int(output.split("\n")[-1])
    body = "\n".join(output.split("\n")[:-1])
    return status_code, body


def curl_post(url: str, headers: dict = None, json_data: dict = None, timeout: int = 60) -> tuple[int, str]:
    cmd = ["curl", "-s", "-X", "POST", url, "-w", "\\n%{http_code}", "--noproxy", "*"]
    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
    if json_data:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(json_data)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = r.stdout.strip()
    status_code = int(output.split("\n")[-1])
    body = "\n".join(output.split("\n")[:-1])
    return status_code, body


@pytest.fixture
def api_key():
    return "test-master-key"


class TestProxyHealth:
    @pytest.mark.integration
    def test_proxy_responds(self, litellm_proxy):
        status, body = curl_get(f"{litellm_proxy['base_url']}/", timeout=5)
        assert status != 0

    @pytest.mark.integration
    def test_proxy_health_endpoint(self, litellm_proxy, api_key):
        status, body = curl_get(
            f"{litellm_proxy['base_url']}/health",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert status == 200

    @pytest.mark.integration
    def test_proxy_models_endpoint(self, litellm_proxy, api_key):
        status, body = curl_get(
            f"{litellm_proxy['base_url']}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert status == 200
        data = json.loads(body)
        assert "data" in data
        models = [m["id"] for m in data.get("data", [])]
        assert "test-model" in models


class TestChatCompletions:
    @pytest.mark.integration
    def test_basic_completion_succeeds(self, litellm_proxy, api_key, mock_api_control):
        mock_api_control.rate_limited_models.discard("gpt-4o-mini")
        status, body = curl_post(
            f"{litellm_proxy['base_url']}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json_data={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        data = json.loads(body)
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


class TestRateLimitHandling:
    @pytest.mark.integration
    def test_rate_limit_detected_from_429_response(self, per_test_proxy, api_key, mock_api_control):
        mock_api_control.rate_limited_models.add("gpt-4o-mini")

        status, body = curl_post(
            f"{per_test_proxy['base_url']}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json_data={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert status == 429, f"Expected 429 from mock, got {status}: {body}"

    @pytest.mark.integration
    def test_requests_succeed_when_not_rate_limited(self, per_test_proxy, api_key, mock_api_control):
        mock_api_control.rate_limited_models.discard("gpt-4o-mini")

        for i in range(3):
            status, body = curl_post(
                f"{per_test_proxy['base_url']}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json_data={"model": "test-model", "messages": [{"role": "user", "content": f"Request {i}"}]},
            )
            assert status == 200, f"Request {i} should succeed, got {status}: {body}"
            data = json.loads(body)
            assert "choices" in data
