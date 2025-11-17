"""Tests for CLI"""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reVReports.cli import main


def test_main(cli_runner):
    """Test main() CLI command"""
    result = cli_runner.invoke(main, "--help")
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )


@pytest.mark.parametrize("command", ["plots", "maps"])
def test_bad_config(cli_runner, command, test_data_dir, caplog, tmp_path):
    """
    Test that the plots and maps command both exit with an error when
    passed an invalid configuration file.
    """
    config_path = test_data_dir / "config_misplaced_params.json"

    result = cli_runner.invoke(
        main,
        [
            command,
            "-c",
            config_path.as_posix(),
            "-o",
            tmp_path.as_posix(),
            "--dpi",
            100,
        ],
    )
    assert result.exit_code == 1, "Command did not fail as expected."
    assert "Input configuration file failed. Exiting process" in caplog.text, (
        "Config validation error message not logged."
    )


def test_unpack_characterizations(cli_runner, test_data_dir, tmp_path):
    """Integration test for unpack_characterizations() CLI command"""

    char_csv = (
        test_data_dir
        / "supply_curves"
        / "characterizations"
        / "supply-curve.csv"
    )
    char_map_path = test_data_dir / "characterization-map.json"

    output_csv = tmp_path / "characterizations.csv"
    result = cli_runner.invoke(
        main,
        [
            "unpack-characterizations",
            "-i",
            char_csv.as_posix(),
            "-m",
            char_map_path.as_posix(),
            "-o",
            output_csv.as_posix(),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )

    output_df = pd.read_csv(output_csv)

    expected_results_src = (
        test_data_dir
        / "outputs"
        / "characterizations"
        / "unpacked-supply-curve.csv"
    )
    expected_df = pd.read_csv(expected_results_src)

    assert_frame_equal(output_df, expected_df)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
