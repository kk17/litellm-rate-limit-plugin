from litellm_rate_limit import RateLimitCallback

rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=1.0,
    probe_models_by_provider={"openai": ["gpt-4o-mini"]},
)
