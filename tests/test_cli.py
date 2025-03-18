# -*- coding: utf-8 -*-
"""Tests for CLI"""
import tempfile
from pathlib import Path

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reVReports.cli import main
from reVReports.plots import compare_images_approx
from reVReports.data import check_files_match


def test_main(cli_runner):
    """Test main() CLI command."""
    result = cli_runner.invoke(main)
    assert result.exit_code == 0, f"Command failed with error {result.exception}"


@pytest.mark.parametrize("command", ["plots", "maps"])
def test_bad_config(cli_runner, command, data_dir, caplog):
    """
    Test that the plots and maps command both exit with an error when passed an invalid
    configuration file.
    """
    config_path = data_dir.joinpath("config_misplaced_params.json")
    with tempfile.TemporaryDirectory() as temp_dir:
        out_path = Path(temp_dir)
        result = cli_runner.invoke(
            main,
            [
                command,
                "-c",
                config_path.as_posix(),
                "-o",
                out_path.as_posix(),
                "--dpi",
                100,
            ],
        )
        assert result.exit_code == 1, "Command did not fail as expected."
        assert (
            "Input configuration file failed. Exiting process" in caplog.text
        ), "Config validation error message not logged."


@pytest.mark.parametrize(
    "tech",
    ["osw", "pv", "wind_bespoke", "wind_vanilla", "pv_site_lcoe_only", "geo"],
)
def test_plots_integration(cli_runner, data_dir, tech, set_to_data_dir):
    # pylint: disable=unused-argument
    """Integration test for the plots command."""

    config_path = data_dir.joinpath(f"config_{tech}.json")
    with tempfile.TemporaryDirectory() as temp_dir:
        out_path = Path(temp_dir)
        result = cli_runner.invoke(
            main,
            [
                "plots",
                "-c",
                config_path.as_posix(),
                "-o",
                out_path.as_posix(),
                "--dpi",
                100,
            ],
        )
        assert result.exit_code == 0, f"Command failed with error {result.exception}"

        test_path = data_dir.joinpath("outputs", "plots", tech)

        patterns = ["*.png", "*.csv"]
        for pattern in patterns:
            outputs_match, difference = check_files_match(pattern, out_path, test_path)
            if not outputs_match:
                raise AssertionError(
                    "Output files do not match expected files. "
                    f"Difference is: {difference}"
                )

        # check outputs were created correctly
        output_image_names = [f.relative_to(out_path) for f in out_path.rglob("*.png")]
        for output_image_name in output_image_names:
            output_image = out_path.joinpath(output_image_name)
            test_image = test_path.joinpath(output_image_name)
            images_match, pct_diff = compare_images_approx(
                output_image, test_image, hash_size=64, max_diff_pct=0.15
            )
            assert images_match, (
                f"{output_image_name} does match expected image. "
                f"Percent difference is: {round(pct_diff * 100, 2)}."
            )

        # check outputs were created correctly
        output_csv_names = [f.relative_to(out_path) for f in out_path.rglob("*.csv")]
        for output_csv_name in output_csv_names:
            output_csv = out_path.joinpath(output_csv_name)
            test_csv = test_path.joinpath(output_csv_name)
            output_df = pd.read_csv(output_csv)
            test_df = pd.read_csv(test_csv)
            assert_frame_equal(output_df, test_df)


@pytest.mark.parametrize(
    "config_name",
    [
        "config_osw.json",
        "config_pv.json",
        "config_wind_bespoke.json",
        "config_wind_bespoke_4_scen.json",
        "config_wind_bespoke_2_scen.json",
        "config_wind_bespoke_1_scen.json",
        "config_wind_vanilla.json",
        "config_pv_map_vars.json",
        "config_geo.json",
    ],
)
def test_maps_integration(cli_runner, data_dir, config_name, set_to_data_dir):
    # pylint: disable=unused-argument
    """Integration test for the maps command."""

    config_path = data_dir.joinpath(config_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        out_path = Path(temp_dir)
        result = cli_runner.invoke(
            main,
            [
                "maps",
                "-c",
                config_path.as_posix(),
                "-o",
                out_path.as_posix(),
                "--dpi",
                100,
            ],
        )
        assert result.exit_code == 0, f"Command failed with error {result.exception}"

        test_folder = config_name.replace("config_", "").replace(".json", "")
        test_path = data_dir.joinpath("outputs", "maps", test_folder)

        pattern = "*.png"
        outputs_match, difference = check_files_match(pattern, out_path, test_path)
        if not outputs_match:
            raise AssertionError(
                "Output files do not match expected files. "
                f"Difference is: {difference}"
            )

        # check outputs were created correctly
        output_image_names = [f.relative_to(out_path) for f in out_path.rglob(pattern)]
        for output_image_name in output_image_names:
            output_image = out_path.joinpath(output_image_name)
            test_image = test_path.joinpath(output_image_name)
            images_match, pct_diff = compare_images_approx(
                output_image, test_image, hash_size=64, max_diff_pct=0.15
            )
            assert images_match, (
                f"{output_image_name} does match expected image. "
                f"Percent difference is: {round(pct_diff * 100, 2)}."
            )


def test_unpack_characterizations(
    cli_runner,
    data_dir,
):
    """Integration test for unpack_characterizations() CLI command."""

    char_csv = data_dir.joinpath(
        "supply_curves", "characterizations", "supply-curve.csv"
    )
    char_map_path = data_dir.joinpath("characterization-map.json")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_csv = Path(temp_dir).joinpath("characterizations.csv")
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
        assert result.exit_code == 0, f"Command failed with error {result.exception}"

        output_df = pd.read_csv(output_csv)

        expected_results_src = data_dir.joinpath(
            "outputs", "characterizations", "unpacked-supply-curve.csv"
        )
        expected_df = pd.read_csv(expected_results_src)

        assert_frame_equal(output_df, expected_df)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
