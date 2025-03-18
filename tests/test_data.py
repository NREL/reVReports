# -*- coding: utf-8 -*-
"""Tests for data module"""
import tempfile
from pathlib import Path

import pytest

from reVReports.data import check_files_match


def test_check_files_match():
    """
    Unit test for the check_files_match() function -- check that it works as expected
    when files match and do not match, exercising the file pattern filter.
    """

    mock_files = [f"{i}.csv" for i in range(0, 5)]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        dir_1_path = temp_dir_path.joinpath("one")
        dir_1_path.mkdir()

        dir_2_path = temp_dir_path.joinpath("two")
        dir_2_path.mkdir()

        for mock_file in mock_files:
            dir_1_path.joinpath(mock_file).touch()
            dir_2_path.joinpath(mock_file).touch()

        # add an extra file with a different extension to dir 1
        dir_1_path.joinpath("mock.png").touch()

        # when filtering to just the txt files, the files should match
        files_match, difference = check_files_match("*.txt", dir_1_path, dir_2_path)
        assert files_match is True and not difference

        # remove the extension filter and check that the files do not match
        files_match, difference = check_files_match("*", dir_1_path, dir_2_path)
        assert files_match is False and difference == [Path("mock.png")]


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
