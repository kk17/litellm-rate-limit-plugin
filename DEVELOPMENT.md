# Development Guide

## Quick Start

### Prerequisites

- Python 3.10+
- uv (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd litellm-rate-limit-plugin

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/litellm_rate_limit --cov-report=html

# Run specific test file
pytest tests/test_parser.py -v

# Run specific test class
pytest tests/test_parser.py::TestExtractAnthropicResetHeader -v
```

### Run Linting

```bash
# Run ruff (linter + formatter)
ruff check src/ tests/
ruff format src/ tests/

# Run pre-commit on all files
pre-commit run --all-files
```

## Project Structure

```
litellm-rate-limit-plugin/
├── src/litellm_rate_limit/
│   ├── __init__.py           # Public exports
│   ├── parser.py             # Rate limit header parsing
│   ├── callback.py           # LiteLLM CustomLogger callback
│   ├── alias_aware_state.py  # Alias-aware health tracking
│   ├── health_state.py       # Health state manager
│   └── health_checker.py     # Background health checks
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_parser.py        # Parser unit tests
│   ├── test_callback.py      # Callback tests
│   ├── test_alias_aware_state.py
│   ├── test_health_state.py
│   └── test_health_checker.py
├── pyproject.toml            # Package configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
├── AGENTS.md                 # Architecture knowledge base
└── DEVELOPMENT.md            # This file
```

## Coding Standards

### Code Style

- Line length: 110 characters
- Python version: 3.10+
- Use type hints for public APIs
- Follow PEP 8 conventions (enforced by ruff)

### Docstrings

Public APIs require docstrings. Use Google-style:

```python
def extract_rate_limit_reset_seconds(error: Exception, default: float = 60.0) -> float:
    """Calculate seconds until rate limit resets.

    Args:
        error: Exception containing response headers.
        default: Default seconds to return if no reset time found.

    Returns:
        Seconds until reset, with fallback to default.
    """
```

### Imports

Imports are sorted automatically by isort with the following config:
- Profile: black
- Line length: 110

### Async Patterns

All state mutations use asyncio locks for thread safety:

```python
async def mark_rate_limited(self, model_id: str, seconds: float) -> None:
    async with self._lock:
        self._rate_limited[model_id] = time.monotonic() + seconds
```

## Testing Guidelines

### Test Organization

- One test file per source file
- Test classes grouped by functionality
- Descriptive test names: `test_<function>_<scenario>_<expected>`

### Fixtures

Shared fixtures are in `tests/conftest.py`:

```python
@pytest.fixture
def mock_rate_limit_error():
    """Create a mock rate limit error with headers."""
    def _create_error(status_code: int = 429, headers: dict = None):
        ...
    return _create_error
```

### Async Tests

Use `pytest-asyncio` with `asyncio_mode = "auto"`:

```python
@pytest.mark.asyncio
async def test_mark_rate_limited():
    manager = HealthStateManager()
    await manager.mark_rate_limited("model-id", 60.0)
    assert await manager.is_rate_limited("model-id")
```

### Time-Dependent Tests

Use `freezegun` for time-dependent tests:

```python
from freezegun import freeze_time

def test_retry_after_seconds():
    with freeze_time("2024-01-01 12:00:00", tz_offset=0):
        # Test code here
```

## Pre-Commit Hooks

Hooks run automatically on commit:

| Hook | Stage | Description |
|------|-------|-------------|
| check-yaml | commit | Validate YAML syntax |
| end-of-file-fixer | commit | Add newline at EOF |
| trailing-whitespace | commit | Remove trailing whitespace |
| isort | commit | Sort imports |
| ruff-check | commit | Lint and auto-fix |
| ruff-format | commit | Format code |
| pylint | pre-push | Additional linting |
| unit tests | manual | Run pytest |

Run manually:

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff-check --all-files

# Run manual hooks
pre-commit run unit-tests
```

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md` with changes
3. Run full test suite: `pytest tests/ -v`
4. Run pre-commit: `pre-commit run --all-files`
5. Build package: `uv build`
6. Tag release: `git tag v0.x.x`

## Architecture Decisions

### Why Monotonic Time?

System clock changes (NTP sync, manual adjustment) can cause:
- Negative durations if clock moves forward
- Infinite waits if clock moves backward

Monotonic time (`time.monotonic()`) guarantees positive, increasing values.

### Why Dataclasses?

- No ORM overhead for simple state tracking
- Type safety with mypy
- Easy serialization if needed later
- Python 3.10+ native support

### Why Post-Hoc Header Parsing?

LiteLLM's callback system doesn't allow intercepting responses before cooldown is set. We parse headers from the exception object after the fact and update the cooldown cache with precise TTL.

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("litellm_rate_limit").setLevel(logging.DEBUG)
```

### Inspect State

```python
# Health state
manager = HealthStateManager()
status = await manager.get_model_status("model-id")
print(status)

# Rate limited models
limited = await manager.get_rate_limited_models()
print(limited)

# Alias-aware state
state = AliasAwareHealthState(router=router)
all_limited = await state.get_all_rate_limited()
print(all_limited)
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Headers not parsed | Exception type doesn't expose headers | Check error object structure |
| Models not restored | Monotonic time drift | Check system clock, use `clear_rate_limit()` |
| Tests fail with time errors | Timezone issues | Use `freeze_time` with `tz_offset=0` |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run: `pre-commit run --all-files`
5. Run: `pytest tests/ -v`
6. Commit with descriptive message
7. Push and create pull request
