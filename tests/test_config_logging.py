"""Tests for config parsing of request logging and file logging options."""

from unittest.mock import patch

import yaml

from litellm_rate_limit.config import (
    FileRotatingConfig,
    LoggingConfig,
    load_config,
)


class TestLoggingConfig:
    def test_defaults(self):
        config = LoggingConfig()
        assert config.log_request_header is False
        assert config.log_request_body is False
        assert config.log_file == ""
        assert config.error_log_file == ""
        assert config.log_file_enabled is False
        assert isinstance(config.log_file_rotating, FileRotatingConfig)

    def test_request_logging_fields(self):
        config = LoggingConfig(log_request_header=True, log_request_body=True)
        assert config.log_request_header is True
        assert config.log_request_body is True


class TestFileRotatingConfig:
    def test_defaults(self):
        config = FileRotatingConfig()
        assert config.handler == "none"
        assert config.max_bytes == 5 * 1024 * 1024
        assert config.backup_count == 3
        assert config.when == "midnight"
        assert config.interval == 1

    def test_custom_values(self):
        config = FileRotatingConfig(
            handler="RotatingFileHandler",
            max_bytes=10 * 1024 * 1024,
            backup_count=5,
        )
        assert config.handler == "RotatingFileHandler"
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5


class TestLoadConfigRequestLogging:
    def test_parse_request_logging_from_yaml(self, tmp_path):
        config_data = {
            "rate_limit_plugin": {
                "logging": {
                    "log_level": "INFO",
                    "log_request_header": True,
                    "log_request_body": True,
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("litellm_rate_limit.config._find_config_file", return_value=config_file):
            config = load_config()

        assert config.logging.log_request_header is True
        assert config.logging.log_request_body is True

    def test_request_logging_defaults(self, tmp_path):
        config_data = {
            "rate_limit_plugin": {
                "logging": {
                    "log_level": "DEBUG",
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("litellm_rate_limit.config._find_config_file", return_value=config_file):
            config = load_config()

        assert config.logging.log_request_header is False
        assert config.logging.log_request_body is False


class TestLoadConfigFileLogging:
    def test_parse_file_logging_from_yaml(self, tmp_path):
        config_data = {
            "rate_limit_plugin": {
                "logging": {
                    "log_file_enabled": True,
                    "log_file": "./logs/app.log",
                    "error_log_file": "./logs/error.log",
                    "log_file_rotating": {
                        "handler": "RotatingFileHandler",
                        "max_bytes": 10485760,
                        "backup_count": 5,
                    },
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("litellm_rate_limit.config._find_config_file", return_value=config_file):
            config = load_config()

        assert config.logging.log_file_enabled is True
        assert config.logging.log_file == "./logs/app.log"
        assert config.logging.error_log_file == "./logs/error.log"
        assert config.logging.log_file_rotating.handler == "RotatingFileHandler"
        assert config.logging.log_file_rotating.max_bytes == 10485760
        assert config.logging.log_file_rotating.backup_count == 5

    def test_file_logging_defaults(self, tmp_path):
        config_data = {
            "rate_limit_plugin": {
                "logging": {
                    "log_level": "DEBUG",
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("litellm_rate_limit.config._find_config_file", return_value=config_file):
            config = load_config()

        assert config.logging.log_file_enabled is False
        assert config.logging.log_file == ""
        assert config.logging.log_file_rotating.handler == "none"

    def test_time_rotation_parsing(self, tmp_path):
        config_data = {
            "rate_limit_plugin": {
                "logging": {
                    "log_file_enabled": True,
                    "log_file": "./logs/app.log",
                    "log_file_rotating": {
                        "handler": "TimedRotatingFileHandler",
                        "when": "H",
                        "interval": 6,
                        "backup_count": 10,
                    },
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("litellm_rate_limit.config._find_config_file", return_value=config_file):
            config = load_config()

        rc = config.logging.log_file_rotating
        assert rc.handler == "TimedRotatingFileHandler"
        assert rc.when == "H"
        assert rc.interval == 6
        assert rc.backup_count == 10
