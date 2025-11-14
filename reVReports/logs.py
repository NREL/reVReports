"""Functionality related to loggers and logging"""

import logging
import sys


LOG_FORMAT = (
    "%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s"
)


def get_logger(name, log_level):
    """Get a logger with the user specified name and log level

    Logger will only include a StreamHandler to sys.stdout.

    Parameters
    ----------
    name : str
        Name of logger.
    log_level : int
        Level of logger: e.g., logging.INFO, logging.DEBUG, etc.

    Returns
    -------
    logging.Logger
        Logger with a StreamHandler to sys.stdout
    """

    logger = logging.getLogger(name)
    logformat = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logformat)
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger
