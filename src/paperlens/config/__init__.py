"""Configuration: typed parameters, YAML loading with `extends`, and explain-config."""

from paperlens.config.loader import load_config
from paperlens.config.models import AppConfig

__all__ = ["AppConfig", "load_config"]
