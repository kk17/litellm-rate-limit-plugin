# LiteLLM Rate Limit Plugin

A plugin for LiteLLM Proxy that provides intelligent health checking and rate limit handling.

## Features

- **Periodic health benchmarking** - Proactive model health checks with latency/quality scoring
- **Rate limit extraction** - Parse exact reset times from API headers (Anthropic, OpenAI)
- **Smart model blocking** - Temporarily remove rate-limited models from routing until reset
- **Automatic restoring** - Unblock models when rate limit window expires
- **Model aliases** - User-friendly alias config with rate limit integration

## Installation

```bash
pip install litellm-rate-limit-plugin
```

## Usage

### Basic Configuration

Add the callback to your LiteLLM config:

```yaml
litellm_settings:
  callbacks: ["rate_limit_callback"]
```

### Programmatic Usage

```python
from litellm_rate_limit import (
    RateLimitCallback,
    HealthStateManager,
    AliasAwareHealthState,
)

# Create callback with custom settings
callback = RateLimitCallback(default_cooldown_seconds=60.0)

# Use health state manager
health_state = HealthStateManager()

# Track rate limits with alias support
alias_state = AliasAwareHealthState(router=your_router)
```

## License

MIT
