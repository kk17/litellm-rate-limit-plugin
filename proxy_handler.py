"""LiteLLM Proxy callback handler for rate limit plugin.

This module exports a pre-configured callback instance for use with LiteLLM Proxy.

Usage in config.yaml:
    litellm_settings:
      callbacks: proxy_handler.rate_limit_callback
"""

from litellm_rate_limit import RateLimitCallback

# Pre-configured callback instance for LiteLLM Proxy
rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=60.0,
)
