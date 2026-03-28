"""Structured logging configuration for the popoly package."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the ``popoly`` package.

    Sets up a :class:`~logging.StreamHandler` on *stderr* with a consistent
    format: ``timestamp | level | module | message``.

    Args:
        level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Returns:
        The ``popoly`` package logger.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    logger = logging.getLogger("popoly")
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on repeated calls.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Update existing handler levels on reconfiguration.
        for handler in logger.handlers:
            handler.setLevel(numeric_level)

    return logger
