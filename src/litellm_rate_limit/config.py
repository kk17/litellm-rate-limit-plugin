"""Configuration loader for rate limit plugin.

Reads configuration from:
1. Environment variables (highest priority)
2. config.yaml's rate_limit_plugin section
3. Default values

Environment variables:
- RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS
- RATE_LIMIT_PROBE_MODELS (JSON dict)
- RATE_LIMIT_HEALTH_CHECK_ENABLED
- RATE_LIMIT_PROVIDER_COOLDOWN (JSON dict)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckConfig:
    """Health check configuration nested under rate_limit_plugin.health_check."""

    enabled: bool = False
    interval_seconds: int = 60
    test_prompt: str = "Say 'ok'"
    max_latency_ms: float = 30000.0
    timeout_seconds: int = 30
    probe_models_by_provider: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class HeaderParsingConfig:
    """Header parsing configuration nested under rate_limit_plugin.header_parsing."""

    enabled: bool = True
    anthropic_reset_header: str = "anthropic-ratelimit-unified-reset"


@dataclass
class RateLimitPluginConfig:
    """Main configuration class for rate limit plugin.

    Maps from config.yaml structure to callback-friendly parameters.

    Config.yaml structure:
        rate_limit_plugin:
          default_cooldown_seconds: 60
          header_parsing:
            enabled: true
            anthropic_reset_header: "anthropic-ratelimit-unified-reset"
          health_check:
            enabled: true
            interval_seconds: 3600
            test_prompt: "Say 'ok'"
            max_latency_ms: 30000
            timeout_seconds: 30
            probe_models_by_provider:
              minimax: ["minimax-m2"]
              zai: ["glm-4.5-air"]
    """

    default_cooldown_seconds: float = 60.0
    provider_cooldown_seconds: dict[str, float] = field(default_factory=dict)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    header_parsing: HeaderParsingConfig = field(default_factory=HeaderParsingConfig)


DEFAULT_CONFIG = RateLimitPluginConfig()


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


def load_config() -> RateLimitPluginConfig:
    """Load configuration with priority: env vars > config.yaml > defaults."""
    config = RateLimitPluginConfig()

    yaml_config = _load_config_from_yaml()

    # Top-level keys
    if "default_cooldown_seconds" in yaml_config:
        config.default_cooldown_seconds = float(yaml_config["default_cooldown_seconds"])

    # Provider cooldown (top-level key in config, not nested)
    if "provider_cooldown_seconds" in yaml_config:
        pc = yaml_config["provider_cooldown_seconds"]
        if isinstance(pc, dict):
            config.provider_cooldown_seconds = {k: float(v) for k, v in pc.items()}

    # Header parsing (nested under header_parsing)
    if "header_parsing" in yaml_config:
        hp = yaml_config["header_parsing"]
        if isinstance(hp, dict):
            if "enabled" in hp:
                config.header_parsing.enabled = bool(hp["enabled"])
            if "anthropic_reset_header" in hp:
                config.header_parsing.anthropic_reset_header = str(hp["anthropic_reset_header"])

    # Health check (nested under health_check)
    if "health_check" in yaml_config:
        hc = yaml_config["health_check"]
        if isinstance(hc, dict):
            if "enabled" in hc:
                config.health_check.enabled = bool(hc["enabled"])
            if "interval_seconds" in hc:
                config.health_check.interval_seconds = int(hc["interval_seconds"])
            if "test_prompt" in hc:
                config.health_check.test_prompt = str(hc["test_prompt"])
            if "max_latency_ms" in hc:
                config.health_check.max_latency_ms = float(hc["max_latency_ms"])
            if "timeout_seconds" in hc:
                config.health_check.timeout_seconds = int(hc["timeout_seconds"])
            if "probe_models_by_provider" in hc:
                config.health_check.probe_models_by_provider = _parse_probe_models(
                    hc["probe_models_by_provider"]
                )

    # Environment variable overrides
    env_cooldown = os.environ.get("RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS")
    if env_cooldown is not None:
        try:
            config.default_cooldown_seconds = float(env_cooldown)
            logger.debug("Override default_cooldown_seconds from env: %s", env_cooldown)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS: %s", e)

    env_health_enabled = os.environ.get("RATE_LIMIT_HEALTH_CHECK_ENABLED")
    if env_health_enabled is not None:
        config.health_check.enabled = env_health_enabled.lower() in ("true", "1", "yes")
        logger.debug("Override health_check.enabled from env: %s", config.health_check.enabled)

    env_interval = os.environ.get("RATE_LIMIT_HEALTH_CHECK_INTERVAL_SECONDS")
    if env_interval is not None:
        try:
            config.health_check.interval_seconds = int(env_interval)
            logger.debug("Override health_check.interval_seconds from env: %s", env_interval)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse RATE_LIMIT_HEALTH_CHECK_INTERVAL_SECONDS: %s", e)

    env_prompt = os.environ.get("RATE_LIMIT_HEALTH_CHECK_PROMPT")
    if env_prompt is not None:
        config.health_check.test_prompt = env_prompt

    env_max_latency = os.environ.get("RATE_LIMIT_HEALTH_CHECK_MAX_LATENCY_MS")
    if env_max_latency is not None:
        try:
            config.health_check.max_latency_ms = float(env_max_latency)
            logger.debug("Override health_check.max_latency_ms from env: %s", env_max_latency)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse RATE_LIMIT_HEALTH_CHECK_MAX_LATENCY_MS: %s", e)

    env_probe_models = os.environ.get("RATE_LIMIT_PROBE_MODELS")
    if env_probe_models:
        parsed = _parse_probe_models(env_probe_models)
        if parsed:
            config.health_check.probe_models_by_provider = parsed

    env_provider_cooldown = os.environ.get("RATE_LIMIT_PROVIDER_COOLDOWN")
    if env_provider_cooldown:
        try:
            parsed = json.loads(env_provider_cooldown)
            if isinstance(parsed, dict):
                config.provider_cooldown_seconds = {k: float(v) for k, v in parsed.items()}
                logger.debug(
                    "Override provider_cooldown_seconds from env: %s", config.provider_cooldown_seconds
                )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to parse RATE_LIMIT_PROVIDER_COOLDOWN: %s", e)

    return config
