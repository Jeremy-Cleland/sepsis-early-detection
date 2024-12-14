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

    This logger supports:
    - Colored console output with different colors for each log level
    - File logging with rotation
    - Optional JSON formatting
    - Customizable date-time format (default: MM-DD HH:MM)
    - Multiple logging handlers (console and file)

    Parameters
    ----------
    name : str
        The name of the logger instance. Used to identify different logger instances
        in a hierarchical manner. Default is "sepsis_prediction".
    log_file : str
        Path to the log file where logs will be written. Directory will be created
        if it doesn't exist. Default is "logs/sepsis_prediction.log".
    level : Union[str, int]
        The logging level. Can be string ('DEBUG', 'INFO', etc.) or corresponding
        integer values. Default is "INFO".
    use_json : bool
        If True, logs will be formatted as JSON objects. Useful for log parsing
        and analysis. Default is False.
    formatter : str
        The format string for log messages when not using JSON format.
        Default is "%(asctime)s - %(message)s".
    max_bytes : int
        Maximum size of each log file in bytes before rotation occurs.
        Default is 5MB (5 * 1024 * 1024 bytes).
    backup_count : int
        Number of backup files to keep when rotating logs.
        Default is 5 files.
    console : bool
        Whether to enable console (stdout) logging in addition to file logging.
        Default is True.

    Returns
    -------
    logging.Logger
        Configured logger instance with specified handlers and formatters.

    Raises
    ------
    ValueError
        If an invalid logging level is provided.
    Exception
        If there's an error during logger setup (falls back to basic configuration).

    Examples
    --------
    >>> logger = setup_logger(
    ...     name="my_app",
    ...     level="DEBUG",
    ...     use_json=True
    ... )
    >>> logger.info("Application started")
    >>> logger.error("An error occurred")
    """
    logging.getLogger().handlers = []
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
    """Remove all handlers from the root logger and other existing loggers."""
    # Remove handlers from root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Disable propagation to root logger for your custom logger
    logging.getLogger("sepsis_prediction").propagate = False
