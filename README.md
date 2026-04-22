# LiteLLM Rate Limit Plugin

A plugin for LiteLLM Proxy that provides intelligent health checking and rate limit handling.

## Features

- **Periodic health benchmarking** - Proactive model health checks with latency/quality scoring
- **Rate limit extraction** - Parse exact reset times from API headers (Anthropic, OpenAI)
- **Smart model blocking** - Temporarily remove rate-limited models from routing until reset
- **Automatic restoring** - Unblock models when rate limit window expires
- **Model aliases** - User-friendly alias config with rate limit integration
- **Model health status sharing** - Map models to share health status with a probe model

## Installation

```bash
pip install litellm-rate-limit-plugin
```

## Usage

### Basic Configuration

1. Add the `rate_limit_plugin` section to your `config.yaml`:

```yaml
litellm_settings:
  callbacks: proxy_handler.rate_limit_callback

rate_limit_plugin:
  default_cooldown_seconds: 60.0
  models_to_check:
    - MiniMax-M2:
        - MiniMax-M2
        - MiniMax-M2.5
    - glm-4.5-air:
        - glm-4.5-air
        - glm-4.5
```

2. Ensure `proxy_handler.py` is in the same directory as your config.yaml (installed with the package).

**Configuration Priority**: Environment variables > config.yaml > defaults

**Environment Variables**:
- `RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS`
- `RATE_LIMIT_PROBE_MODELS` (JSON dict)
- `RATE_LIMIT_HEALTH_CHECK_ENABLED`

### Fallback Configuration

Configure model fallbacks when rate limits are hit:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4-turbo
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-opus-20240229
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229

router_settings:
  # Enable fallback routing
  num_retries: 3
  retry_after: 5

  # Fallback to other deployments of same model
  fallbacks: [
    {
      "gpt-4": ["gpt-4", "claude-3"]
    },
    {
      "claude-3": ["claude-3", "gpt-4"]
    }
  ]

litellm_settings:
  callbacks: ["rate_limit_callback"]
```

### Health Check Configuration

Enable proactive health checking to detect issues before requests fail:

```yaml
litellm_settings:
  callbacks: ["rate_limit_callback"]

rate_limit_plugin:
  # Default cooldown when no reset time in headers
  default_cooldown_seconds: 60

  # Header parsing configuration
  header_parsing:
    enabled: true
    # Anthropic's unified reset header (Unix timestamp)
    anthropic_reset_header: "anthropic-ratelimit-unified-reset"

  # Health check configuration
  health_check:
    enabled: true
    # How often to run health checks (seconds)
    interval_seconds: 60
    # Prompt to send for health checks
    test_prompt: "Say 'ok'"
    # Maximum acceptable latency (ms)
    max_latency_ms: 30000
    # Timeout for individual health checks (seconds)
    timeout_seconds: 30
```

### Model Health Status Sharing

Configure which models get health-checked and share health status. Each entry specifies a probe model (key) and the list of models (values) that share its health status. Models not listed are NOT health-checked (status: unknown):

```yaml
rate_limit_plugin:
  health_check:
    # Specify model mappings for health status sharing
    # The key model serves as the probe, values share its health status
    # Only probe models (keys) are health-checked
    models_to_check:
      - MiniMax-M2:
          - MiniMax-M2
          - MiniMax-M2.5
      - glm-4.5-air:
          - glm-4.5-air
          - glm-4.5
      - grok-code-fast-1:
          - grok-code-fast-1
          - gpt-5-mini
```

**How it works:**

| Config | Probe Model | Shared Models | Behavior |
|--------|-------------|---------------|----------|
| `- MiniMax-M2: [MiniMax-M2, MiniMax-M2.5]` | `MiniMax-M2` | `MiniMax-M2`, `MiniMax-M2.5` | Both share `MiniMax-M2` health status |
| `- glm-4.5-air: [glm-4.5-air, glm-4.5]` | `glm-4.5-air` | `glm-4.5-air`, `glm-4.5` | Both share `glm-4.5-air` health status |
| (not listed) | N/A | N/A | NOT health-checked (status: unknown) |

**Example:**

```yaml
models_to_check:
  - glm-4.5-air:
      - glm-4.5-air
      - glm-4.5
```

- `glm-4.5-air` is the probe model (health-checked)
- All models in the list (`glm-4.5-air`, `glm-4.5`) share `glm-4.5-air`'s health status
- Models not listed anywhere are NOT health-checked (status: unknown)

**Programmatic Usage:**

```python
from litellm_rate_limit import HealthStateManager, ProviderProbeConfig

# Configure model health status sharing
probe_config = ProviderProbeConfig(
    models_to_check=[
        {"MiniMax-M2": ["MiniMax-M2", "MiniMax-M2.5"]},
        {"glm-4.5-air": ["glm-4.5-air", "glm-4.5"]},
    ]
)

# Create health manager with probe config
health_state = HealthStateManager(provider_probe_config=probe_config)

# Mark probe model as rate-limited
await health_state.mark_rate_limited("MiniMax-M2", 60.0)

# All models sharing MiniMax-M2 health status are now blocked
assert await health_state.is_rate_limited("MiniMax-M2.5") is True
```

### Programmatic Usage

```python
import asyncio
from litellm_rate_limit import (
    RateLimitCallback,
    HealthStateManager,
    HealthCheckRunner,
    HealthBenchmark,
    AliasAwareHealthState,
    ProviderProbeConfig,
)

async def main():
    # Configure model health status sharing
    probe_config = ProviderProbeConfig(
        models_to_check=[
            {"MiniMax-M2": ["MiniMax-M2", "MiniMax-M2.5"]},
            {"glm-4.5-air": ["glm-4.5-air", "glm-4.5"]},
        ]
    )

    # Create callback with custom settings
    callback = RateLimitCallback(default_cooldown_seconds=60.0)

    # Set the router reference for cooldown updates
    callback.set_router(litellm_router)

    # Health state manager with probe config
    health_state = HealthStateManager(provider_probe_config=probe_config)

    # Mark a model as rate-limited
    await health_state.mark_rate_limited(
        model_id="anthropic/claude-3-opus",
        seconds_until_reset=120.0,
    )

    # Check if model is rate-limited (auto-restores when expired)
    if await health_state.is_rate_limited("anthropic/claude-3-opus"):
        print("Model is rate-limited, use fallback")

    # Get all healthy models from a list
    all_models = ["gpt-4", "claude-3-opus", "claude-3-sonnet"]
    healthy = await health_state.get_healthy_models(all_models)
    print(f"Available models: {healthy}")

    # Alias-aware state (integrates with model_group_alias)
    alias_state = AliasAwareHealthState(router=litellm_router)
    await alias_state.mark_rate_limited("claude-opus", 60.0)
    # Both "claude-opus" and its target will be blocked

    # Background health checker
    benchmark = HealthBenchmark(
        test_prompt="Say 'ok'",
        timeout_seconds=30.0,
        max_latency_ms=30000.0,
    )

    runner = HealthCheckRunner(benchmark=benchmark)

    # Start periodic health checks
    await runner.start_periodic_checks(
        name="primary-check",
        models=["gpt-4", "claude-3-opus"],
        interval_seconds=60,
        health_manager=health_state,
        client=your_llm_client,  # Optional: custom client
    )

    # Later: stop health checks
    await runner.stop_all()

asyncio.run(main())
```

### Integration with LiteLLM Proxy

```python
# In your LiteLLM proxy startup
from litellm_rate_limit import RateLimitCallback, HealthStateManager

# Create shared health state
health_state = HealthStateManager()

# Create callback with reference to health state
callback = RateLimitCallback(
    default_cooldown_seconds=60.0,
)

# The callback will automatically:
# 1. Detect 429 errors from API responses
# 2. Parse reset time from headers (Anthropic, OpenAI, standard)
# 3. Update LiteLLM's cooldown_cache with precise TTL
```

### Supported Rate Limit Headers

The plugin extracts reset times from these headers (in priority order):

| Header | Format | Provider |
|--------|--------|----------|
| `anthropic-ratelimit-unified-reset` | Unix timestamp | Anthropic |
| `retry-after` | HTTP-date or seconds | Standard |
| `x-ratelimit-reset` | Seconds | OpenAI-style |

If no header is found, falls back to `default_cooldown_seconds` (default: 60s).

## License

MIT
