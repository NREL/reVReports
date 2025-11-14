"""reVReports command line interface"""

import logging
from pathlib import Path
import sys
import json

import click
import pandas as pd
from pydantic import ValidationError
from matplotlib import font_manager

from reVReports import __version__
from reVReports.configs import Config
from reVReports import characterizations
from reVReports.fonts import SANS_SERIF, SANS_SERIF_BOLD
from reVReports import logs
from reVReports.plots import PlotData, PlotGenerator, make_bar_plot
from reVReports.maps import MapData, MapGenerator, configure_map_params
from reVReports.utilities.plots import configure_matplotlib, DPI

font_manager.fontManager.ttflist.extend([SANS_SERIF, SANS_SERIF_BOLD])

LOGGER = logs.get_logger(__name__, "INFO")
MAX_NUM_SCENARIOS = 4

configure_matplotlib()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Flag to turn on debug logging. Default is not verbose.",
)
@click.pass_context
def main(ctx, verbose):
    """reVReports command line interface"""
    ctx.ensure_object(dict)
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)


@main.command()
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to input configuration JSON file.",
)
@click.option(
    "--out-path",
    "-o",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help=(
        "Path to output folder where plots will be saved. "
        "Folder will be created if it does not exist."
    ),
)
@click.option(
    "--dpi",
    "-d",
    required=False,
    type=int,
    default=DPI,
    help=f"Resolution of output images in dots per inch. Default is {DPI}.",
)
def plots(config_file, out_path, dpi):
    """Create report plots for configured supply curves"""

    config = _load_config(config_file)

    # make output directory (only if needed)
    out_path.mkdir(parents=False, exist_ok=True)

    plot_data = PlotData(config)
    _display_summary_statistics(plot_data)
    _summarize_state_level_results(plot_data.all_df, out_path)

    make_bar_plot(
        data_df=plot_data.top_level_sums_df,
        y_col="capacity_gw",
        ylabel="Capacity (GW)",
        scenario_palette=config.scenario_palette,
        out_image_path=out_path / "total_capacity.png",
        dpi=dpi,
    )

    make_bar_plot(
        data_df=plot_data.top_level_sums_df,
        y_col="area_developable_sq_km",
        ylabel="Developable Area (sq. km.)",
        scenario_palette=config.scenario_palette,
        out_image_path=out_path / "total_area.png",
        dpi=dpi,
    )

    plotter = PlotGenerator(plot_data, out_path, dpi)
    plotter.build_supply_curves()
    plotter.build_capacity_by_region_bar_chart()
    plotter.build_transmission_box_plots()
    plotter.build_box_plots()
    plotter.build_histograms()
    plotter.build_regional_box_plots()

    LOGGER.info("Command completed successfully.")


@main.command()
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to input configuration JSON file.",
)
@click.option(
    "--out-path",
    "-o",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help=(
        "Path to output folder where plots will be saved. "
        "Folder will be created if it does not exist."
    ),
)
@click.option(
    "--dpi",
    "-d",
    required=False,
    type=int,
    default=DPI,
    help=f"Resolution of output images in dots per inch. Default is {DPI}.",
)
def maps(config_file, out_path, dpi):
    """Create report maps for configured supply curves"""

    config = _load_config(config_file)

    n_scenarios = len(config.scenarios)
    if n_scenarios > MAX_NUM_SCENARIOS:
        LOGGER.error("Cannot map more than %d scenarios.", MAX_NUM_SCENARIOS)
        sys.exit(1)

    out_path.mkdir(parents=False, exist_ok=True)

    cap_col, point_size, map_vars = configure_map_params(config)

    map_data = MapData(config, cap_col=cap_col)
    plotter = MapGenerator(map_data)
    plotter.build_maps(map_vars, out_path, dpi, point_size=point_size)

    LOGGER.info("Command completed successfully.")


@main.command()
@click.option(
    "--supply_curve_csv",
    "-i",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to bespoke wind supply curve CSV file created by reV",
)
@click.option(
    "--char_map",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to JSON file storing characterization map",
)
@click.option(
    "--out_csv",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to CSV to store results",
)
@click.option(
    "--cell_size",
    "-c",
    required=False,
    default=90.0,
    type=float,
    help=("Cell size in meters of characterization layers. Default is 90."),
)
def unpack_characterizations(
    supply_curve_csv, char_map, out_csv, cell_size=90.0
):
    """Unpack characterization data from the input supply curve

    The unpacking converts values from embedded JSON strings to new
    standalone columns, and saves a new version of the supply curve with
    these columns included.
    """

    LOGGER.info("Loading supply curve data")
    supply_curve_df = pd.read_csv(supply_curve_csv)

    LOGGER.info("Loading characterization mapping")
    with Path(char_map).open("r", encoding="utf-8") as f:
        characterization_map = json.load(f)

    LOGGER.info("Unpacking characterizations")
    char_df = characterizations.unpack_characterizations(
        supply_curve_df, characterization_map, cell_size
    )

    char_df.to_csv(out_csv, header=True, index=False)
    LOGGER.info("Command completed successfully.")


def _load_config(config_file):
    """Load user configuration from disk"""

    LOGGER.info("Starting plot creation")
    LOGGER.info("Loading configuration file %s", config_file)
    try:
        config = Config.from_json(config_file)
    except ValidationError:
        LOGGER.exception("Input configuration file failed. Exiting process.")
        sys.exit(1)
    LOGGER.info("Configuration file loaded.")

    n_scenarios = len(config.scenarios)
    LOGGER.info("%d supply curve scenarios will be plotted:", n_scenarios)
    for scenario in config.scenarios:
        LOGGER.info("\t%s: %s", scenario.name, scenario.source.name)

    return config


def _display_summary_statistics(plot_data):
    """Log high level scenario metrics"""

    LOGGER.info("Summary statistics:")

    sum_area_by_scenario_md = plot_data.top_level_sums_df[
        ["Scenario", "area_developable_sq_km"]
    ].to_markdown(tablefmt="rounded_grid", floatfmt=",.0f")
    LOGGER.info("\nDevelopable Area:\n%s", sum_area_by_scenario_md)

    sum_cap_by_scenario_md = plot_data.top_level_sums_df[
        ["Scenario", "capacity_gw"]
    ].to_markdown(tablefmt="rounded_grid", floatfmt=",.1f")
    LOGGER.info("\nCapacity:\n%s", sum_cap_by_scenario_md)

    sum_aep_by_scenario_md = plot_data.top_level_sums_df[
        ["Scenario", "aep_twh"]
    ].to_markdown(tablefmt="rounded_grid", floatfmt=",.1f")
    LOGGER.info("\nGeneration:\n%s", sum_aep_by_scenario_md)


def _summarize_state_level_results(all_df, out_path):
    """Write state level summary statistics"""

    LOGGER.info("Summarizing state level results")
    all_df["cf_x_area"] = (
        all_df["capacity_factor"] * all_df["area_developable_sq_km"]
    )
    state_sum_df = all_df.groupby(["Scenario", "state"], as_index=False)[
        [
            "area_developable_sq_km",
            "capacity_mw",
            "annual_energy_site_mwh",
            "cf_x_area",
        ]
    ].sum()
    state_sum_df["area_wt_mean_cf"] = (
        state_sum_df["cf_x_area"] / state_sum_df["area_developable_sq_km"]
    )
    state_sum_df = state_sum_df.drop(columns=["cf_x_area"])
    state_sum_df = state_sum_df.sort_values(
        by=["state", "Scenario"], ascending=True
    )
    out_csv = out_path / "state_results_summary.csv"
    LOGGER.info("Saving state level results to %s", out_csv)
    state_sum_df.to_csv(out_csv, header=True, index=False)
