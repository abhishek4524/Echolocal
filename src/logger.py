"""
src/logger.py
Centralized logging configuration for EchoLocal.
All modules import from here instead of using print() directly.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Return a module-scoped logger with consistent formatting.

    Args:
        name: Module name — pass __name__ from the calling module.
        log_level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler — UTF-8 so Hindi characters render correctly on Windows
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    try:
        handler.stream.reconfigure(encoding="utf-8")  # Python 3.7+
    except AttributeError:
        pass

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # Optional file handler — writes to project_root/logs/echolocal.log
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "echolocal.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Non-fatal — console logging still works

    return logger


def set_global_log_level(level: str) -> None:
    """
    Adjust log level for ALL echolocal loggers at runtime.
    Useful for --verbose CLI flag.

    Args:
        level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    for name, lgr in logging.Logger.manager.loggerDict.items():
        if isinstance(lgr, logging.Logger):
            lgr.setLevel(numeric)
            for h in lgr.handlers:
                h.setLevel(numeric)
