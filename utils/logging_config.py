#utils/logging_config.py

import logging
from typing import Optional

def setup_logging(log_level: Optional[int] = logging.INFO, log_format: Optional[str] = None):
    """
    Sets up the logging configuration for the application

    This function configures the logging system with the specified log level and log format.
    By default, the log level is set to INFO, and the format to INFO, and the format includes timestamp,
    log level, and the message. 

    Args:
        log_level (int, optional): Logging level (e.g. logging.DEBUG, logging.INFO, etc).  
        log_format (str, optional): Log message format. Default includes timestamp with log level, message, 
    """
    # Set a default format if none is provided
    if log_format is None:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=log_level,
        format=log_format
    )