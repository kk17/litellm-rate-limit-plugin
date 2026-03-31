# AGENTS.md - LiteLLM Rate Limit Plugin Knowledge Base

## Project Overview

A LiteLLM Proxy plugin providing intelligent health checking and rate limit handling. Work as a non-invasive callback plugin.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LiteLLM Proxy                               │
├─────────────────────────────────────────────────────────────────┤
│  Request Flow:                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Request  │───▶│ Pre-Call     │───▶│ Model Selection     │   │
│  │          │    │ Hook         │    │ (Router)            │   │
│  └──────────┘    └──────────────┘    └─────────────────────┘   │
│                         │                      │                │
│                         ▼                      ▼                │
│                  ┌──────────────┐    ┌─────────────────────┐   │
│                  │ Health State │◀───│ Model API Call      │   │
│                  │ Manager      │    │                     │   │
│                  └──────────────┘    └─────────────────────┘   │
│                         │                      │                │
│                         ▼                      ▼                │
│                  ┌──────────────┐    ┌─────────────────────┐   │
│                  │ Rate Limit   │◀───│ Post-Failure Hook   │   │
│                  │ Callback     │    │ (429 Detection)     │   │
│                  └──────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Parser (`src/litellm_rate_limit/parser.py`)

**Purpose**: Extract rate limit reset times from API error headers.

**Key Functions**:
- `is_rate_limit_error(error)` - Detect 429 errors
- `extract_rate_limit_reset_dt(error)` - Parse reset datetime from headers
- `extract_rate_limit_reset_seconds(error)` - Get seconds until reset

**Supported Headers** (in priority order):
1. `anthropic-ratelimit-unified-reset` - Unix timestamp (Anthropic)
2. `retry-after` - HTTP-date or seconds (Standard)
3. `x-ratelimit-reset` - Seconds (OpenAI-style)

### 2. Callback (`src/litellm_rate_limit/callback.py`)

**Purpose**: LiteLLM CustomLogger integration for rate limit handling.

**Class**: `RateLimitCallback(CustomLogger)`

**Hooks Used**:
- `async_post_call_failure_hook` - Detect 429, extract reset time, update cooldown
- `async_pre_call_hook` - Filter out rate-limited models before routing

**Router Initialization**:
- `_ensure_router()` lazily obtains router from `litellm.proxy.proxy_server.llm_router`
- **Auto-starts health checks** when router is first obtained (idempotent via `_health_checks_started` flag)
- No explicit `set_router()` call required - plugin self-initializes on first hook invocation

**Integration Point**: Updates `router.cooldown_cache` with precise TTL.

### 3. Alias-Aware State (`src/litellm_rate_limit/alias_aware_state.py`)

**Purpose**: Track rate limits with model alias resolution.

**Class**: `AliasAwareHealthState`

**Key Behavior**:
- Resolves aliases via `router.model_group_alias`
- Stores rate limits by target model name
- All aliases blocked when target is rate-limited

### 4. Health State (`src/litellm_rate_limit/health_state.py`)

**Purpose**: Central state manager for model health tracking.

**Class**: `HealthStateManager`

**Features**:
- Monotonic time for reliable duration tracking
- Auto-restore expired rate limits on access
- Track consecutive failures and last errors

### 5. Health Checker (`src/litellm_rate_limit/health_checker.py`)

**Purpose**: Background health benchmarking.

**Classes**:
- `HealthBenchmark` - Run individual health checks
- `HealthCheckRunner` - Manage periodic background tasks

**Metrics**: Latency (ms), response validity, error tracking

### 6. Provider Probe Config (`src/litellm_rate_limit/provider_probe.py`)

**Purpose**: Share health status across models from the same provider.

**Class**: `ProviderProbeConfig`

**Key Behavior**:
- Maps model prefixes to probe models
- First model in config list is the probe model
- Explicitly listed models get their own health status
- Unlisted models share the probe model's health status

**Example**:
```yaml
probe_models_by_provider:
  minimax: ["minimax-m2"]
  zai: ["glm-4.5-air", "glm-5"]
```

- `minimax-m2` health status used for ALL `minimax-*` models
- `glm-4.5-air` used for unlisted `glm-*` models
- `glm-5` has its own independent health status

## Data Flows

### Rate Limit Detection Flow

```
429 Error Detected
       │
       ▼
┌──────────────────┐
│ Parser extracts  │
│ reset time from  │
│ headers          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Callback updates │
│ cooldown_cache   │
│ with precise TTL │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Health State     │
│ marks model as   │
│ rate-limited     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Next request     │
│ skips model      │
│ until reset      │
└──────────────────┘
```

### Auto-Restore Flow

```
is_rate_limited(model) called
            │
            ▼
    ┌───────────────┐
    │ Check if      │
    │ model tracked │
    └───────┬───────┘
            │
       ┌────┴────┐
       │         │
    No │         │ Yes
       ▼         ▼
  Return     ┌───────────────┐
  False      │ monotonic()   │
             │ >= until?     │
             └───────┬───────┘
                     │
                ┌────┴────┐
                │         │
             No │         │ Yes
                ▼         ▼
           Return     Remove entry
           True       Return False
```

## Configuration Schema

```yaml
litellm_settings:
  callbacks: ["rate_limit_callback"]

rate_limit_plugin:
  default_cooldown_seconds: 60
  header_parsing:
    enabled: true
    anthropic_reset_header: "anthropic-ratelimit-unified-reset"
  health_check:
    enabled: true
    interval_seconds: 60
    test_prompt: "Say 'ok'"
    max_latency_ms: 30000
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use monotonic time | Avoid issues with clock adjustments |
| Parse headers post-hoc | Can't intercept before LiteLLM sets cooldown |
| Track by deployment ID | Same model_name can have multiple deployments |
| Dataclass-based | Simple, typed, no ORM overhead |
| AsyncIO locks | Thread-safe state mutations |

## Error Handling Patterns

```python
# Header parsing fallback chain
anthropic_header → retry_after → x_ratelimit_reset → default (60s)

# Negative/past reset times
if seconds_until_reset <= 0:
    return default_cooldown_seconds

# Missing cooldown_cache
if not hasattr(router, 'cooldown_cache'):
    log_warning()
    return  # Graceful degradation
```

## Testing Strategy

| Component | Test Type | Tools |
|-----------|-----------|-------|
| Parser | Unit tests | pytest, freezegun |
| Callback | Async unit tests | pytest-asyncio, Mock |
| Health State | Concurrency tests | asyncio.gather |
| Health Checker | Integration tests | pytest-asyncio |

## Integration Points

### With LiteLLM Router

```python
# Router reference is obtained lazily via _ensure_router()
# No explicit set_router() call needed - plugin auto-initializes

# Access cooldown cache
router.cooldown_cache.set_cooldown(model_id, cooldown_time)

# Access alias mapping
router.model_group_alias.get(alias_name, default)
```

**Health Check Auto-Start**: When `_ensure_router()` first obtains the router reference, it automatically starts health checks if enabled. This is idempotent - multiple calls won't start duplicate health checks.

### With Custom Logger Base

```python
class RateLimitCallback(CustomLogger):
    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ) -> None:
        ...
```

## Common Patterns

### Adding a New Header Parser

1. Add parsing function in `parser.py`
2. Update `extract_rate_limit_reset_dt()` to try new header
3. Add tests in `test_parser.py`

### Extending Health State

1. Add field to `ModelHealthStatus` dataclass
2. Update `mark_rate_limited()` and `is_rate_limited()`
3. Add corresponding tests

## Known Limitations

1. **Streaming responses**: Rate limits detected mid-stream handled on next request
2. **Multi-instance**: Health state not shared (future: Redis integration)
3. **Pre-call filtering**: Relies on LiteLLM's existing cooldown mechanism
