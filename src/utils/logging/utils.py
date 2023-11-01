"""Logging utils methods"""
import logging
from typing import Optional


def get_logger(log_level: int = logging.INFO, name: Optional[str] = __name__) -> logging.Logger:
    """Add logger instance into module

    Args:
        log_level (int, optional): Logger's messages level. Defaults to logging.INFO.
        name (Optional[str], optional): Name of the model. Defaults to __name__.

    Returns:
        logging.Logger: _description_
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    return logger
