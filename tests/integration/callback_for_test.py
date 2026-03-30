from litellm_rate_limit import RateLimitCallback

rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=2.0,
    provider_cooldown_seconds={"github-copilot": 300.0, "minimax": 30.0},
    probe_models_by_provider={"openai": ["primary-model"]},
)
