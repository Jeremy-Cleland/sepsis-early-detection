# src/logger_config.py

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Union

import colorlog
from pythonjsonlogger import jsonlogger

# Global dict to track configured loggers
_CONFIGURED_LOGGERS = {}


def setup_logger(
    name: str = "sepsis_prediction",
    log_file: str = "logs/sepsis_prediction.log",
    level: Union[str, int] = "INFO",
    use_json: bool = False,
    formatter: str = "%(asctime)s - %(message)s",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with both console and file handling capabilities.
    """
    # Check if logger was already configured
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    try:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Handle logging level configuration
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
            if not isinstance(level, int):
                raise ValueError(f"Invalid log level: {level}")

        # Initialize logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Prevent propagation to ancestor loggers

        # Remove any existing handlers
        logger.handlers.clear()

        # Configure console logging if enabled
        if console:
            console_handler = colorlog.StreamHandler()
            console_handler.setLevel(level)

            if use_json:
                console_formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s"
                )
            else:
                console_formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(message)s%(reset)s",
                    datefmt="%m-%d %H:%M",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={},
                    style="%",
                )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)

        if use_json:
            file_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
        else:
            file_formatter = logging.Formatter(formatter, datefmt="%m-%d %H:%M")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Store the configured logger
        _CONFIGURED_LOGGERS[name] = logger

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        fallback_logger = logging.getLogger(name)
        fallback_logger.error(f"Failed to set up logger '{name}': {e}")
        return fallback_logger

    return logger


# Add function to get existing logger
def get_logger(
    name: str = "sepsis_prediction",
    level: Union[str, int] = "INFO",
    use_json: bool = False,
) -> logging.Logger:
    """Get an existing logger or create a new one with custom settings."""
    if name in _CONFIGURED_LOGGERS:
        logger = _CONFIGURED_LOGGERS[name]
        logger.setLevel(level)  # Update level even if logger exists
        return logger

    return setup_logger(
        name=name, level=level, use_json=use_json, log_file=f"logs/{name}.log"
    )


def disable_duplicate_logging():
    """Disable duplicate logging by removing handlers from the root logger."""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
