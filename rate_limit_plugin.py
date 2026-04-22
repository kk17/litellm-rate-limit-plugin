"""LiteLLM Proxy callback handler for rate limit plugin.

DEPLOYMENT - Bridge File Pattern (REQUIRED):

    LiteLLM's callback loader only supports files relative to config.yaml.
    It does NOT support loading callbacks from installed packages.

    Create a bridge file (e.g., callback.py) in the SAME directory as config.yaml:

        # callback.py - bridge file
        from litellm_rate_limit import RateLimitCallback
        from litellm_rate_limit.config import load_config

        config = load_config()
        rate_limit_callback = RateLimitCallback.from_config(config)

    In config.yaml:

        litellm_settings:
          callbacks: ["callback.rate_limit_callback"]

        rate_limit_plugin:
          default_cooldown_seconds: 60.0
          health_check:
            enabled: true
            models_to_check:
              - minimax-m2:
                - minimax-m2
                - minimax-m2.5

    This file (rate_limit_plugin.py) can be used directly if copied to config directory:

        litellm_settings:
          callbacks: ["rate_limit_plugin.callback"]
"""

from litellm_rate_limit import RateLimitCallback
from litellm_rate_limit.config import load_config

config = load_config()
# print(f"Loaded config: {config}")
callback = RateLimitCallback.from_config(config)
