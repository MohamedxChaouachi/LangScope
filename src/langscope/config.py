"""
config.py — Start-up configuration.

The only setting is the log level, read from the LANGSCOPE_LOGLEVEL environment variable
(DEBUG, INFO, WARNING, ERROR or CRITICAL; INFO if it is unset or not a level name).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

LOG_LEVEL_VARIABLE = "LANGSCOPE_LOGLEVEL"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass(frozen=True)
class Config:
    log_level: int = logging.INFO


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    """Read the configuration from *environ* (the process environment by default)."""
    env = os.environ if environ is None else environ
    name = env.get(LOG_LEVEL_VARIABLE, "INFO").strip().upper()
    # getLevelName returns the number for a level name, and a "Level X" string for anything else.
    level = logging.getLevelName(name)
    return Config(log_level=level if isinstance(level, int) else logging.INFO)


def configure_logging(config: Config) -> None:
    """Send log records to stderr at the configured level."""
    logging.basicConfig(level=config.log_level, format=LOG_FORMAT)
