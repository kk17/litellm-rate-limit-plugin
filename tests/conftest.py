"""Shared pytest fixtures for litellm-rate-limit-plugin tests."""

import pytest


@pytest.fixture
def mock_rate_limit_error():
    """Create a mock rate limit error with headers."""

    def _create_error(status_code: int = 429, headers: dict = None):
        error = type("MockError", (), {})()
        error.status_code = status_code
        error.headers = headers or {}
        return error

    return _create_error


@pytest.fixture
def mock_router():
    """Create a mock LiteLLM router."""

    def _create_router(model_group_alias: dict = None):
        router = type("MockRouter", (), {})()
        router.model_group_alias = model_group_alias or {}
        return router

    return _create_router
