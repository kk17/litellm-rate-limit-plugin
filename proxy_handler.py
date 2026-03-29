"""LiteLLM Proxy callback handler for rate limit plugin.

Usage in config.yaml:
    litellm_settings:
      callbacks: proxy_handler.rate_limit_callback
"""

from litellm_rate_limit import RateLimitCallback

rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=60.0,
    probe_models_by_provider={
        "minimax": ["minimax-m2"],
        "glm": ["glm-4.5-air", "glm-5"],
    },
)
