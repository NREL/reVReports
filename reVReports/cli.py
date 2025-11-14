"""reVReports command line interface"""

import logging
from pathlib import Path
import warnings
import sys
import json

import click
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import numpy as np
import tqdm
from pydantic import ValidationError
from matplotlib import pyplot as plt
from matplotlib import font_manager

from reVReports import __version__
from reVReports.configs import Config, VALID_TECHS
from reVReports.data import augment_sc_df
from reVReports.utilities.maps import (
    map_geodataframe_column,
    DEFAULT_BOUNDARIES,
)
from reVReports import characterizations
from reVReports.fonts import SANS_SERIF, SANS_SERIF_BOLD
from reVReports import logs
from reVReports.plots import PlotData, PlotGenerator, make_bar_plot
from reVReports.utilities.plots import (
    configure_matplotlib,
    DPI,
    SMALL_SIZE,
    BIGGER_SIZE,
    RC_FONT_PARAMS,
)
from reVReports.exceptions import reVReportsValueError

font_manager.fontManager.ttflist.extend([SANS_SERIF, SANS_SERIF_BOLD])

WIND = {"wind", "osw"}
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
    """reVReports command line interface."""
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
    """Create standard set of report plots for input supply curves"""

    config, n_scenarios = _load_config(config_file)

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
    """Create standard set of report maps for input supply curves"""

    config, n_scenarios = _load_config(config_file)

    # make output directory (only if needed)
    out_path.mkdir(parents=False, exist_ok=True)

    LOGGER.info("Loading boundaries dataset")
    boundaries_gdf = gpd.read_file(DEFAULT_BOUNDARIES)
    boundaries_gdf.to_crs("EPSG:4326", inplace=True)
    boundaries_single_part_gdf = boundaries_gdf.explode(index_parts=True)
    boundaries_dissolved = boundaries_gdf.union_all()
    center_lon = boundaries_dissolved.centroid.x
    center_lat = boundaries_dissolved.centroid.y
    background_gdf = gpd.GeoDataFrame(
        {"geometry": [boundaries_dissolved]}, crs=boundaries_gdf.crs
    ).explode(index_parts=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        map_extent = background_gdf.buffer(0.01).total_bounds

    LOGGER.info("Configuring map settings")
    map_vars = {
        config.lcoe_all_in_col: {
            "breaks": [25, 30, 35, 40, 45, 50, 60, 70],
            "cmap": "YlGn",
            "legend_title": "All-in LCOE ($/MWh)",
        },
        config.lcoe_site_col: {
            "breaks": [25, 30, 35, 40, 45, 50, 60, 70],
            "cmap": "YlGn",
            "legend_title": "Project LCOE ($/MWh)",
        },
        "lcot_usd_per_mwh": {
            "breaks": [5, 10, 15, 20, 25, 30, 40, 50],
            "cmap": "YlGn",
            "legend_title": "LCOT ($/MWh)",
        },
        "area_developable_sq_km": {
            "breaks": [5, 10, 25, 50, 100, 120],
            "cmap": "BuPu",
            "legend_title": "Developable Area (sq km)",
        },
    }

    cf_col = config.cf_col or "capacity_factor_ac"

    point_size = 2.0
    if config.tech == "pv":
        cap_col = "capacity_dc_mw"
        map_vars.update(
            {
                "capacity_dc_mw": {
                    "breaks": [100, 500, 1000, 2000, 3000, 4000],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity DC (MW)",
                },
                "capacity_ac_mw": {
                    "breaks": [100, 500, 1000, 2000, 3000, 4000],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity AC (MW)",
                },
                cf_col: {
                    "breaks": [0.2, 0.25, 0.3, 0.35],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity Factor",
                },
            }
        )
    elif config.tech == "wind":
        cap_col = "capacity_ac_mw"
        map_vars.update(
            {
                "capacity_ac_mw": {
                    "breaks": [60, 120, 180, 240, 275],
                    "cmap": "Blues",
                    "legend_title": "Capacity (MW)",
                },
                "capacity_density": {
                    "breaks": [2, 3, 4, 5, 6, 10],
                    "cmap": "Blues",
                    "legend_title": "Capacity Density (MW/sq km)",
                },
                cf_col: {
                    "breaks": [0.25, 0.3, 0.35, 0.4, 0.45],
                    "cmap": "Blues",
                    "legend_title": "Capacity Factor",
                },
                "losses_wakes_pct": {
                    "breaks": [6, 7, 8, 9, 10],
                    "cmap": "Purples",
                    "legend_title": "Wake Losses (%)",
                },
            }
        )
    elif config.tech == "osw":
        point_size = 1.5
        cap_col = "capacity_ac_mw"
        map_vars.update(
            {
                "capacity_ac_mw": {
                    "breaks": [200, 400, 600, 800, 1000],
                    "cmap": "PuBu",
                    "legend_title": "Capacity (MW)",
                },
                "capacity_density": {
                    "breaks": [0.5, 1, 2, 3, 5, 10],
                    "cmap": "PuBu",
                    "legend_title": "Capacity Density (MW/sq km)",
                },
                cf_col: {
                    "breaks": [0.3, 0.35, 0.4, 0.45, 0.5],
                    "cmap": "PuBu",
                    "legend_title": "Capacity Factor",
                },
                "area_developable_sq_km": {
                    "breaks": [10, 50, 100, 200, 225, 250],
                    "cmap": "BuPu",
                    "legend_title": "Developable Area (sq km)",
                },
                config.lcoe_all_in_col: {
                    "breaks": [100, 125, 150, 175, 200],
                    "cmap": "YlGn",
                    "legend_title": "All-in LCOE ($/MWh)",
                },
                config.lcoe_site_col: {
                    "breaks": [75, 100, 125, 150, 175, 200],
                    "cmap": "YlGn",
                    "legend_title": "Project LCOE ($/MWh)",
                },
                "lcot_usd_per_mwh": {
                    "breaks": [15, 20, 25, 30, 35, 40, 50, 60],
                    "cmap": "YlGn",
                    "legend_title": "LCOT ($/MWh)",
                },
                "cost_export_usd_per_mw_ac": {
                    "breaks": [
                        500_000,
                        600_000,
                        700_000,
                        800_000,
                        900_000,
                        1_000_000,
                    ],
                    "cmap": "YlGn",
                    "legend_title": "Export Cable ($/MW)",
                },
                "dist_export_km": {
                    "breaks": [50, 75, 100, 125, 150],
                    "cmap": "YlGn",
                    "legend_title": "Export Cable Distance (km)",
                },
                "losses_wakes_pct": {
                    "breaks": [6, 7, 8, 9, 10],
                    "cmap": "Purples",
                    "legend_title": "Wake Losses (%)",
                },
            }
        )
    elif config.tech == "geo":
        cap_col = "capacity_ac_mw"
        map_vars.update(
            {
                "capacity_ac_mw": {
                    "breaks": [200, 400, 600, 800, 1000],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity (MW)",
                },
                "capacity_density": {
                    "breaks": [2, 3, 4, 6, 10, 15],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity Density (MW/sq km)",
                },
                cf_col: {
                    "breaks": [0.99, 0.9925, 0.995, 0.9975, 0.999],
                    "cmap": "YlOrRd",
                    "legend_title": "Capacity Factor",
                },
            }
        )
    else:
        msg = (
            f"Invalid input: tech={config.tech}. Valid options are: "
            f"{VALID_TECHS}"
        )
        raise reVReportsValueError(msg)

    # add/modify map variables based on input config parameters
    for map_var in config.map_vars:
        map_var_data = map_var.model_dump()
        col = map_var_data.pop("column")
        map_vars[col] = map_var_data

    # remove map vars that are in the exclude list
    for exclude_map in config.exclude_maps:
        if exclude_map in map_vars:
            map_vars.pop(exclude_map)

    LOGGER.info("Loading and augmenting supply curve data")
    scenario_dfs = {}
    for scenario in tqdm.tqdm(config.scenarios, total=n_scenarios):
        scenario_df = pd.read_csv(scenario.source)

        # drop zero capacity sites
        scenario_sub_df = scenario_df[scenario_df["capacity_ac_mw"] > 0].copy()

        supply_curve_gdf = gpd.GeoDataFrame(
            scenario_sub_df,
            geometry=gpd.points_from_xy(
                x=scenario_sub_df["longitude"], y=scenario_sub_df["latitude"]
            ),
            crs="EPSG:4326",
        )
        supply_curve_gdf["capacity_density"] = (
            supply_curve_gdf[cap_col]
            / supply_curve_gdf["area_developable_sq_km"].replace(0, np.nan)
        ).replace(np.nan, 0)

        scenario_dfs[scenario.name] = supply_curve_gdf

    n_scenarios = len(config.scenarios)
    if n_scenarios > MAX_NUM_SCENARIOS:
        LOGGER.error("Cannot map more than %d scenarios.", MAX_NUM_SCENARIOS)
        sys.exit(1)

    n_cols = 2
    n_rows = int(np.ceil(n_scenarios / n_cols))
    LOGGER.info("Creating maps")
    for map_var, map_settings in tqdm.tqdm(map_vars.items()):
        with plt.rc_context(RC_FONT_PARAMS):
            fig, ax = plt.subplots(
                ncols=2,
                nrows=n_rows,
                figsize=(13, 4 * n_rows),
                subplot_kw={
                    "projection": gplt.crs.AlbersEqualArea(
                        central_longitude=center_lon,
                        central_latitude=center_lat,
                    )
                },
            )
            for i, scenario_name in enumerate(scenario_dfs):
                scenario_df = scenario_dfs[scenario_name]
                if map_var not in scenario_df.columns:
                    err = (
                        f"{map_var} column not found in one or more input "
                        "supply curves. Consider using the `exclude_maps` "
                        "configuration option to skip map generation for "
                        "this column."
                    )
                    LOGGER.error(err)
                panel = ax.ravel()[i]
                panel = map_geodataframe_column(
                    scenario_df,
                    map_var,
                    color_map=map_settings.get("cmap"),
                    breaks=map_settings.get("breaks"),
                    map_title=None,
                    legend_title=map_settings.get("legend_title"),
                    background_df=background_gdf,
                    boundaries_df=boundaries_single_part_gdf,
                    extent=map_extent,
                    layer_kwargs={
                        "s": point_size,
                        "linewidth": 0,
                        "marker": "o",
                    },
                    legend_kwargs={
                        "marker": "s",
                        "frameon": False,
                        "bbox_to_anchor": (1, 0.5),
                        "loc": "center left",
                    },
                    legend=(i + 1 == n_scenarios),
                    ax=panel,
                )
                panel.patch.set_alpha(0)
                panel.set_title(scenario_name, y=0.88)

            n_panels = len(ax.ravel())
            min_xcoord = -0.04
            mid_xcoord = 0.465
            min_ycoord = 0.0
            mid_ycoord = 0.475
            if n_scenarios in {3, 4}:
                panel_width = 0.6
                panel_height = 0.52
                panel_dims = [panel_width, panel_height]

                lower_lefts = [
                    [mid_xcoord, min_ycoord],
                    [min_xcoord, min_ycoord],
                    [mid_xcoord, mid_ycoord],
                    [min_xcoord, mid_ycoord],
                ]
                for j in range(n_panels):
                    coords = lower_lefts[j]
                    ax.ravel()[-j - 1].set_position(coords + panel_dims)
            elif n_scenarios in {1, 2}:
                ax.ravel()[0].set_position([-0.25, 0.0, 1, 1])
                ax.ravel()[1].set_position([0.27, 0.0, 1, 1])

            if n_scenarios < n_panels:
                extra_panel = ax.ravel()[-1]
                legend_panel_position = extra_panel.get_position()
                fig.delaxes(extra_panel)
                legend_font_size = BIGGER_SIZE
                legend_loc = "center"
                legend_cols = 1
            else:
                legend_font_size = SMALL_SIZE
                legend_loc = "center left"
                legend_cols = 3
                if n_rows == 2:  # noqa: PLR2004
                    legend_panel_position = [
                        mid_xcoord - 0.06,
                        min_ycoord - 0.03,
                        0.2,
                        0.2,
                    ]
                elif n_rows == 1:
                    legend_panel_position = [
                        mid_xcoord - 0.06,
                        min_ycoord + 0.03,
                        0.2,
                        0.2,
                    ]

            legend = fig.axes[-1].get_legend()
            legend_texts = [t.get_text() for t in legend.texts]
            legend_handles = legend.legend_handles
            legend.remove()

            legend_panel = fig.add_subplot(alpha=0, frame_on=False)
            legend_panel.set_axis_off()
            legend_panel.set_position(legend_panel_position)

            legend_panel.legend(
                legend_handles,
                legend_texts,
                frameon=False,
                loc=legend_loc,
                title=map_settings["legend_title"],
                ncol=legend_cols,
                handletextpad=-0.1,
                columnspacing=0,
                fontsize=legend_font_size,
                title_fontproperties={
                    "size": legend_font_size,
                    "weight": "bold",
                },
            )

            out_image_name = f"{map_var}.png"
            out_image_path = out_path / out_image_name
            fig.savefig(out_image_path, dpi=dpi, transparent=True)
            plt.close(fig)

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
    """Unpack characterization data from the input supply curve.

    The unpacking converts values from embedded JSON strings to new
    standalone columns, and saves out a new version of the supply curve
    with these columns included.
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

    return config, n_scenarios


def _load_and_augment_supply_curve_data(config, n_scenarios):
    LOGGER.info("Loading and augmenting supply curve data")
    scenario_dfs = []
    for i, scenario in tqdm.tqdm(
        enumerate(config.scenarios), total=n_scenarios
    ):
        scenario_df = pd.read_csv(scenario.source)

        try:
            aug_df = augment_sc_df(
                scenario_df,
                scenario_name=scenario.name,
                scenario_number=i,
                tech=config.tech,
                lcoe_all_in_col=config.lcoe_all_in_col,
            )
        except KeyError:
            LOGGER.warning(
                "Required columns are missing from the input supply curve. "
                "Was your supply curve created by reV≥v0.14.5?"
            )
            raise

        # drop sites with zero capacity
        # (this also removes inf values for total_lcoe)
        aug_df_w_capacity = aug_df[aug_df["capacity_mw"] > 0].copy()

        scenario_dfs.append(aug_df_w_capacity)

    # combine the data into a single data frame
    all_df = pd.concat(scenario_dfs)
    all_df = all_df.sort_values(
        by=["scenario_number", config.lcoe_all_in_col], ascending=True
    )
    return all_df, scenario_dfs


def _display_summary_statistics(plot_data):
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
