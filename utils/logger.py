"""Lightweight logging utilities (console + optional file)."""
import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "gdad",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or fetch) a configured logger.

    Logs go to stdout and, if ``log_dir`` is given, also to ``log_dir/run.log``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times in the same process.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
