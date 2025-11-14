"""Tests for CLI"""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reVReports.cli import main
from reVReports.utilities.plots import compare_images_approx
from reVReports.data import check_files_match


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


@pytest.mark.parametrize(
    "tech",
    ["osw", "pv", "wind_bespoke", "wind_vanilla", "pv_site_lcoe_only", "geo"],
)
def test_plots_integration(
    cli_runner, test_data_dir, tech, set_to_data_dir, tmp_path
):
    """Integration test for the plots command"""

    config_path = test_data_dir / f"config_{tech}.json"

    result = cli_runner.invoke(
        main,
        [
            "plots",
            "-c",
            config_path.as_posix(),
            "-o",
            tmp_path.as_posix(),
            "--dpi",
            100,
        ],
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )

    test_path = test_data_dir / "outputs" / "plots" / tech

    patterns = ["*.png", "*.csv"]
    for pattern in patterns:
        outputs_match, difference = check_files_match(
            pattern, tmp_path, test_path
        )
        if not outputs_match:
            msg = (
                "Output files do not match expected files. "
                f"Difference is: {difference}"
            )
            raise AssertionError(msg)

    # check outputs were created correctly
    output_image_names = [
        f.relative_to(tmp_path) for f in tmp_path.rglob("*.png")
    ]
    for output_image_name in output_image_names:
        output_image = tmp_path / output_image_name
        test_image = test_path / output_image_name
        images_match, pct_diff = compare_images_approx(
            output_image, test_image, hash_size=64, max_diff_pct=0.15
        )
        assert images_match, (
            f"{output_image_name} does match expected image. "
            f"Percent difference is: {round(pct_diff * 100, 2)}."
        )

    # check outputs were created correctly
    output_csv_names = [
        f.relative_to(tmp_path) for f in tmp_path.rglob("*.csv")
    ]
    for output_csv_name in output_csv_names:
        output_csv = tmp_path / output_csv_name
        test_csv = test_path / output_csv_name
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
def test_maps_integration(
    cli_runner, test_data_dir, config_name, set_to_data_dir, tmp_path
):
    """Integration test for the maps command"""

    config_path = test_data_dir / config_name

    result = cli_runner.invoke(
        main,
        [
            "maps",
            "-c",
            config_path.as_posix(),
            "-o",
            tmp_path.as_posix(),
            "--dpi",
            100,
        ],
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )

    test_folder = config_name.replace("config_", "").replace(".json", "")
    test_path = test_data_dir / "outputs" / "maps" / test_folder

    pattern = "*.png"
    outputs_match, difference = check_files_match(pattern, tmp_path, test_path)
    if not outputs_match:
        msg = (
            "Output files do not match expected files. "
            f"Difference is: {difference}"
        )
        raise AssertionError(msg)

    # check outputs were created correctly
    output_image_names = [
        f.relative_to(tmp_path) for f in tmp_path.rglob(pattern)
    ]
    for output_image_name in output_image_names:
        output_image = tmp_path / output_image_name
        test_image = test_path / output_image_name
        images_match, pct_diff = compare_images_approx(
            output_image, test_image, hash_size=64, max_diff_pct=0.15
        )
        assert images_match, (
            f"{output_image_name} does match expected image. "
            f"Percent difference is: {round(pct_diff * 100, 2)}."
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
