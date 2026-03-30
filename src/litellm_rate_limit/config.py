"""Configuration loader for rate limit plugin.

Reads configuration from:
1. Environment variables (highest priority)
2. config.yaml's rate_limit_plugin section
3. Default values

Environment variables:
- RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS
- RATE_LIMIT_PROBE_MODELS (JSON dict)
- RATE_LIMIT_HEALTH_CHECK_ENABLED
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "default_cooldown_seconds": 60.0,
    "provider_cooldown_seconds": {},
    "probe_models_by_provider": {},
    "health_check_enabled": False,
    "health_check_interval_seconds": 60,
    "health_check_prompt": "Say 'ok'",
    "health_check_max_latency_ms": 30000.0,
}


def _find_config_file() -> Path | None:
    """Find the LiteLLM config file.

    Priority:
    1. LiteLLM proxy's runtime config path (user_config_file_path)
    2. CONFIG_FILE_PATH environment variable (LiteLLM standard)
    3. LITELLM_CONFIG_PATH environment variable
    4. config.yaml in current working directory
    """
    # 1. Try LiteLLM proxy's runtime config path
    try:
        from litellm.proxy.proxy_server import user_config_file_path

        if user_config_file_path:
            path = Path(user_config_file_path)
            if path.exists():
                return path
    except ImportError:
        logger.debug("LiteLLM proxy module not available")

    # 2. Try LiteLLM's standard CONFIG_FILE_PATH env var
    config_path = os.environ.get("CONFIG_FILE_PATH")
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path

    # 3. Try LITELLM_CONFIG_PATH env var
    config_path = os.environ.get("LITELLM_CONFIG_PATH")
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path

    # 4. Try config files in current working directory
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    return None


def _load_config_from_yaml() -> dict[str, Any]:
    """Load rate_limit_plugin section from config.yaml."""
    config_file = _find_config_file()
    if not config_file:
        logger.debug("No config.yaml found, using defaults")
        return {}

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        plugin_config = config.get("rate_limit_plugin", {})
        logger.info("Loaded rate_limit_plugin config from %s", config_file)
        return plugin_config
    except Exception as e:
        logger.warning("Failed to load config from %s: %s", config_file, e)
        return {}


def _parse_probe_models(value: Any) -> dict[str, list[str]]:
    """Parse probe_models_by_provider from various formats."""
    if isinstance(value, dict):
        return {k: list(v) if not isinstance(v, list) else v for k, v in value.items()}

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def load_config() -> dict[str, Any]:
    """Load configuration with priority: env vars > config.yaml > defaults."""
    config = DEFAULT_CONFIG.copy()

    yaml_config = _load_config_from_yaml()
    config.update(yaml_config)

    env_mappings = {
        "RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS": ("default_cooldown_seconds", float),
        "RATE_LIMIT_HEALTH_CHECK_ENABLED": (
            "health_check_enabled",
            lambda x: x.lower() in ("true", "1", "yes"),
        ),
        "RATE_LIMIT_HEALTH_CHECK_INTERVAL_SECONDS": ("health_check_interval_seconds", int),
        "RATE_LIMIT_HEALTH_CHECK_PROMPT": ("health_check_prompt", str),
        "RATE_LIMIT_HEALTH_CHECK_MAX_LATENCY_MS": ("health_check_max_latency_ms", float),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                config[config_key] = converter(value)
                logger.debug("Override %s from env: %s", config_key, config[config_key])
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse %s: %s", env_var, e)

    probe_models_env = os.environ.get("RATE_LIMIT_PROBE_MODELS")
    if probe_models_env:
        parsed = _parse_probe_models(probe_models_env)
        if parsed:
            config["probe_models_by_provider"] = parsed

    provider_cooldown_env = os.environ.get("RATE_LIMIT_PROVIDER_COOLDOWN")
    if provider_cooldown_env:
        try:
            parsed = json.loads(provider_cooldown_env)
            if isinstance(parsed, dict):
                config["provider_cooldown_seconds"] = {k: float(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to parse RATE_LIMIT_PROVIDER_COOLDOWN: %s", e)

    return config
