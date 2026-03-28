"""LiteLLM Rate Limit Plugin.

A plugin for LiteLLM Proxy that provides intelligent health checking and rate limit handling.

Features:
- Periodic health benchmarking with latency/quality scoring
- Rate limit extraction from API headers (Anthropic, OpenAI)
- Smart model blocking with automatic restoration
- Model alias support with rate limit integration
"""

from litellm_rate_limit.alias_aware_state import AliasAwareHealthState, RateLimitEntry
from litellm_rate_limit.callback import RateLimitCallback
from litellm_rate_limit.health_checker import (
    HealthBenchmark,
    HealthCheckResult,
    HealthCheckRunner,
    HealthStatus,
)
from litellm_rate_limit.health_state import HealthStateManager, ModelHealthStatus
from litellm_rate_limit.parser import (
    DEFAULT_COOLDOWN_SECONDS,
    extract_rate_limit_reset_dt,
    extract_rate_limit_reset_seconds,
    is_rate_limit_error,
)

__all__ = [
    "AliasAwareHealthState",
    "DEFAULT_COOLDOWN_SECONDS",
    "HealthBenchmark",
    "HealthCheckResult",
    "HealthCheckRunner",
    "HealthStateManager",
    "HealthStatus",
    "ModelHealthStatus",
    "RateLimitCallback",
    "RateLimitEntry",
    "extract_rate_limit_reset_dt",
    "extract_rate_limit_reset_seconds",
    "is_rate_limit_error",
]

__version__ = "0.1.0"
