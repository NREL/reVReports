"""Custom Exceptions and Errors for reVReports"""

import logging


logger = logging.getLogger("reVReports")


class reVReportsError(Exception):  # noqa: N801
    """Generic reVReports Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class reVReportsValueError(reVReportsError, ValueError):  # noqa: N801
    """reVReports ValueError"""


class reVReportsTypeError(reVReportsError, TypeError):  # noqa: N801
    """reVReports TypeError"""
