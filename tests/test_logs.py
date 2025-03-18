# -*- coding: utf-8 -*-
"""Tests for logs module"""
import logging
import pytest

from reVReports.logs import get_logger


def test_get_logger_happy(caplog):
    """
    Basic unit test for the get_logger() function to assert that it emits messages
    at the correct levels. Formatting is not checked due to apparent limitations of
    caplog fixture.
    """
    logger_name = "test_logger"
    logger = get_logger(logger_name, logging.INFO)

    msgs = [
        (logging.INFO, "Hi there!"),
        (logging.WARNING, "Beware!"),
        (logging.ERROR, "Oh No!"),
    ]
    for level, msg in msgs:
        logger.log(level, msg)

    # check count of messages and records is
    log_messages = caplog.text.rstrip().split("\n")
    log_records = caplog.records
    assert len(log_records) == len(log_messages) == len(msgs)

    # check levels and contens of messages
    for i, msg in enumerate(msgs):
        caplog_record = caplog.record_tuples[i]
        assert caplog_record[0] == logger_name
        assert caplog_record[1] == msg[0]
        assert caplog_record[2] == msg[1]


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
