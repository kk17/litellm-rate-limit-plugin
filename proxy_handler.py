"""LiteLLM Proxy callback handler for rate limit plugin.

Usage in config.yaml:
    litellm_settings:
      callbacks: proxy_handler.rate_limit_callback

    rate_limit_plugin:
      default_cooldown_seconds: 60.0
      probe_models_by_provider:
        minimax: ["minimax-m2"]
        glm: ["glm-4.5-air", "glm-5"]
"""

from litellm_rate_limit import RateLimitCallback
from litellm_rate_limit.config import load_config

_config = load_config()

rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=_config["default_cooldown_seconds"],
    probe_models_by_provider=_config.get("probe_models_by_provider"),
    health_check_enabled=_config.get("health_check_enabled", False),
    health_check_interval_seconds=_config.get("health_check_interval_seconds", 60),
    health_check_prompt=_config.get("health_check_prompt", "Say 'ok'"),
    health_check_max_latency_ms=_config.get("health_check_max_latency_ms", 30000.0),
)
