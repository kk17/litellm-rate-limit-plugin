from litellm_rate_limit import RateLimitCallback

rate_limit_callback = RateLimitCallback(
    default_cooldown_seconds=2.0,
    models_to_check=[{"primary-model": ["primary-model"]}],
)
