# -*- coding: utf-8 -*-
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
import seaborn as sns
from pydantic import ValidationError
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import font_manager


from reVReports import __version__
from reVReports.configs import Config
from reVReports.data import augment_sc_df, ORDERED_REGIONS
from reVReports.maps import map_geodataframe_column, DEFAULT_BOUNDARIES
from reVReports import characterizations
from reVReports.fonts import SANS_SERIF, SANS_SERIF_BOLD
from reVReports import logs
from reVReports.plots import (
    format_graph,
    wrap_labels,
    configure_matplotlib,
    DPI,
    DEFAULT_RC_PARAMS,
    NO_OUTLINE_RC_PARAMS,
    SMALL_SIZE,
    SMALL_MEDIUM_SIZE,
    BIGGER_SIZE,
    RC_FONT_PARAMS,
)

font_manager.fontManager.ttflist.extend([SANS_SERIF, SANS_SERIF_BOLD])

WIND = {"wind", "osw"}
LOGGER = logs.get_logger(__name__, "INFO")

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
    """
    Create standard set of report plots for input supply curves.
    """
    # pylint: disable=too-many-statements, too-many-branches
    LOGGER.info("Starting plot creation")

    LOGGER.info(f"Loading configuration file {config_file}")
    try:
        config = Config.from_json(config_file)
    except ValidationError:
        LOGGER.exception("Input configuration file failed. Exiting process.")
        sys.exit(1)
    LOGGER.info("Configuration file loaded.")

    n_scenarios = len(config.scenarios)
    LOGGER.info(f"{n_scenarios} supply curve scenarios will be plotted:")
    for scenario in config.scenarios:
        LOGGER.info(f"\t{scenario.name}: {scenario.source.name}")

    # make output directory (only if needed)
    out_path.mkdir(parents=False, exist_ok=True)

    # load and augment data
    LOGGER.info("Loading and augmenting supply curve data")
    scenario_dfs = []
    for i, scenario in tqdm.tqdm(enumerate(config.scenarios), total=n_scenarios):
        scenario_df = pd.read_csv(scenario.source)

        try:
            aug_df = augment_sc_df(
                scenario_df,
                scenario_name=scenario.name,
                scenario_number=i,
                tech=config.tech,
                lcoe_all_in_col=config.lcoe_all_in_col,
            )
        except KeyError as e:
            LOGGER.warning(
                "Required columns are missing from the input supply curve. "
                "Was your supply curve created by reV≥v0.14.5?"
            )
            raise e

        # drop sites with zero capacity (this also removes inf values for total_lcoe)
        aug_df_w_capacity = aug_df[aug_df["capacity_mw"] > 0].copy()

        scenario_dfs.append(aug_df_w_capacity)

    # combine the data into a single data frame
    all_df = pd.concat(scenario_dfs)
    all_df.sort_values(
        by=["scenario_number", config.lcoe_all_in_col], ascending=True, inplace=True
    )

    LOGGER.info("Summary statistics:")
    top_level_sums_df = (
        all_df.groupby("Scenario")[
            ["area_developable_sq_km", "capacity_mw", "annual_energy_site_mwh"]
        ]
        .sum()
        .reset_index()
    )
    top_level_sums_df["capacity_gw"] = top_level_sums_df["capacity_mw"] / 1000.0
    top_level_sums_df["aep_twh"] = (
        top_level_sums_df["annual_energy_site_mwh"] / 1000.0 / 1000
    )

    sum_area_by_scenario_md = top_level_sums_df[
        ["Scenario", "area_developable_sq_km"]
    ].to_markdown(tablefmt="rounded_grid", floatfmt=",.0f")
    LOGGER.info(f"\nDevelopable Area:\n{sum_area_by_scenario_md}")

    sum_cap_by_scenario_md = top_level_sums_df[["Scenario", "capacity_gw"]].to_markdown(
        tablefmt="rounded_grid", floatfmt=",.1f"
    )
    LOGGER.info(f"\nCapacity:\n{sum_cap_by_scenario_md}")

    sum_aep_by_scenario_md = top_level_sums_df[["Scenario", "aep_twh"]].to_markdown(
        tablefmt="rounded_grid", floatfmt=",.1f"
    )
    LOGGER.info(f"\nGeneration:\n{sum_aep_by_scenario_md}")

    # summarize state level results by scenario
    LOGGER.info("Summarizing state level results")
    all_df["cf_x_area"] = all_df["capacity_factor"] * all_df["area_developable_sq_km"]
    state_sum_df = all_df.groupby(["Scenario", "state"], as_index=False)[
        ["area_developable_sq_km", "capacity_mw", "annual_energy_site_mwh", "cf_x_area"]
    ].sum()
    state_sum_df["area_wt_mean_cf"] = (
        state_sum_df["cf_x_area"] / state_sum_df["area_developable_sq_km"]
    )
    state_sum_df.drop(columns=["cf_x_area"], inplace=True)
    state_sum_df.sort_values(by=["state", "Scenario"], ascending=True, inplace=True)
    out_csv = out_path.joinpath("state_results_summary.csv")
    LOGGER.info(f"Saving state level results to {out_csv}")
    state_sum_df.to_csv(out_csv, header=True, index=False)

    # create scenario palette for plotting
    scenario_palette = {}
    for scenario in config.scenarios:
        scenario_palette[scenario.name] = scenario.color

    # Barchart showing total capacity by scenario
    LOGGER.info("Plotting total capacity by scenario barchart")
    with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(RC_FONT_PARAMS):
        fig, ax = plt.subplots(figsize=(8, 5))
        y = "capacity_gw"
        ymax = top_level_sums_df[y].max() * 1.05
        g = sns.barplot(
            top_level_sums_df,
            x="Scenario",
            y=y,
            hue="Scenario",
            palette=scenario_palette,
            order=scenario_palette,
            ax=ax,
            legend=False,
        )
        g = format_graph(g, xlabel=None, ylabel="Capacity (GW)", ymax=ymax)
        wrap_labels(g, 10)
        g.set_xticks(g.get_xticks())
        g.set_xticklabels(g.get_xticklabels(), weight="bold")
        out_image_path = out_path.joinpath("total_capacity.png")
        plt.tight_layout()
        g.figure.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    # Barchart showing total developable area by scenario
    LOGGER.info("Plotting total area by scenario barchart")
    with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(RC_FONT_PARAMS):
        fig, ax = plt.subplots(figsize=(8, 5))
        y = "area_developable_sq_km"
        ymax = top_level_sums_df[y].max() * 1.05
        g = sns.barplot(
            top_level_sums_df,
            x="Scenario",
            y=y,
            hue="Scenario",
            palette=scenario_palette,
            order=scenario_palette,
            ax=ax,
            legend=False,
        )
        g = format_graph(g, xlabel=None, ylabel="Developable Area (sq. km.)", ymax=ymax)
        wrap_labels(g, 10)
        g.set_xticks(g.get_xticks())
        g.set_xticklabels(g.get_xticklabels(), weight="bold")
        out_image_path = out_path.joinpath("total_area.png")
        plt.tight_layout()
        fig.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    LOGGER.info("Plotting supply curves")
    # Prepare data for plotting supply curves
    # Set up data frame we can use to plot all-in and site lcoe together on supply curve
    # plots
    supply_curve_total_lcoe_df = all_df[
        [
            config.lcoe_all_in_col,
            "capacity_mw",
            "annual_energy_site_mwh",
            "cumul_capacit_gw",
            "cumul_aep_twh",
            "Scenario",
        ]
    ].copy()
    supply_curve_total_lcoe_df["LCOE Value"] = "All-In LCOE"
    supply_curve_total_lcoe_df.rename(
        columns={config.lcoe_all_in_col: "lcoe"}, inplace=True
    )

    supply_curve_site_lcoe_df = all_df[
        [config.lcoe_site_col, "capacity_mw", "annual_energy_site_mwh", "Scenario"]
    ].copy()
    supply_curve_site_lcoe_df.sort_values(
        by=[config.lcoe_site_col], ascending=True, inplace=True
    )
    supply_curve_site_lcoe_df["cumul_capacit_gw"] = (
        supply_curve_site_lcoe_df.groupby("Scenario")["capacity_mw"].cumsum() / 1000
    )
    supply_curve_site_lcoe_df["cumul_aep_twh"] = (
        supply_curve_site_lcoe_df.groupby("Scenario")["annual_energy_site_mwh"].cumsum()
        / 1000
        / 1000
    )
    supply_curve_site_lcoe_df["LCOE Value"] = "Site LCOE"
    supply_curve_site_lcoe_df.rename(
        columns={config.lcoe_site_col: "lcoe"}, inplace=True
    )

    supply_curve_df = pd.concat([supply_curve_total_lcoe_df, supply_curve_site_lcoe_df])

    # Supply curves - Two panel figure showing Cumulative Capacity by LCOE and
    # Cumulative Generation by LCOE
    if config.lcoe_all_in_col != config.lcoe_site_col:
        sc_line_style = "LCOE Value"
    else:
        sc_line_style = None
    with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
        RC_FONT_PARAMS | {"xtick.labelsize": SMALL_SIZE, "ytick.labelsize": SMALL_SIZE}
    ):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
        panel_1 = sns.lineplot(
            data=supply_curve_df,
            y="lcoe",
            x="cumul_capacit_gw",
            hue="Scenario",
            palette=scenario_palette,
            hue_order=scenario_palette,
            style=sc_line_style,
            ax=ax[0],
        )
        panel_1 = format_graph(
            panel_1,
            xmin=0,
            xmax=None,
            ymin=0,
            ymax=config.plots.site_lcoe_max,
            xlabel="Cumulative Capacity (GW)",
            ylabel="Levelized Cost of Energy ($/MWh)",
            drop_legend=True,
        )
        panel_2 = sns.lineplot(
            data=supply_curve_df,
            y="lcoe",
            x="cumul_aep_twh",
            hue="Scenario",
            palette=scenario_palette,
            hue_order=scenario_palette,
            style=sc_line_style,
            ax=ax[1],
        )
        panel_2 = format_graph(
            panel_2,
            xmin=0,
            xmax=None,
            ymin=0,
            ymax=config.plots.site_lcoe_max,
            xlabel="Cumulative Annual Generation (TWh)",
            ylabel="Levelized Cost of Energy ($/MWh)",
            move_legend_outside=True,
        )
        out_image_path = out_path.joinpath("supply_curves.png")
        plt.tight_layout()
        fig.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    # single panel supply curve - LCOE vs cumulative capacity
    with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(RC_FONT_PARAMS):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax = sns.lineplot(
            data=supply_curve_df,
            y="lcoe",
            x="cumul_capacit_gw",
            hue="Scenario",
            palette=scenario_palette,
            hue_order=scenario_palette,
            style=sc_line_style,
            ax=ax,
        )
        # handles, labels = ax.get_legend_handles_labels()
        ax = format_graph(
            ax,
            xmin=0,
            xmax=None,
            ymin=0,
            ymax=config.plots.site_lcoe_max,
            xlabel="Cumulative Capacity (GW)",
            ylabel="Levelized Cost of Energy ($/MWh)",
            drop_legend=False,
        )
        sns.move_legend(ax, "lower center", ncol=2, fontsize=SMALL_MEDIUM_SIZE)
        out_image_path = out_path.joinpath("supply_curves_capacity_only.png")
        plt.tight_layout()
        fig.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    LOGGER.info("Plotting capacity by region and scenario barchart")
    # Regional capacity comparison
    # Sum the capacity by nrel region
    region_col = "offtake_state" if config.tech == "osw" else "nrel_region"
    econ_cap_by_region_df = (  # noqa
        all_df[all_df[config.lcoe_all_in_col] <= config.plots.total_lcoe_max]
        .groupby(["Scenario", region_col])["capacity_mw"]
        .sum()
        .reset_index()
    )
    econ_cap_by_region_df["capacity_gw"] = econ_cap_by_region_df["capacity_mw"] / 1000
    econ_cap_by_region_df.sort_values("capacity_gw", ascending=False, inplace=True)

    with sns.axes_style("whitegrid", NO_OUTLINE_RC_PARAMS), plt.rc_context(
        RC_FONT_PARAMS
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        g = sns.barplot(
            econ_cap_by_region_df,
            y=region_col,
            x="capacity_gw",
            hue="Scenario",
            dodge=True,
            palette=scenario_palette,
        )
        g = format_graph(g, xlabel="Total Capacity (GW)", ylabel="Region")
        if config.tech == "osw":
            g.set_yticks(g.get_yticks())
            g.set_yticklabels(g.get_yticklabels(), fontsize=10)
        out_image_path = out_path.joinpath("regional_capacity_barchart.png")
        plt.tight_layout()
        fig.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    LOGGER.info("Plotting Transmission Cost and Distance Boxplots")
    for scenario_name, scenario_df in all_df.groupby(["Scenario"], as_index=False):
        # extract transmission cost data in tidy/long format
        if config.tech == "osw":
            trans_cost_vars = {
                "cost_export_usd_per_mw_ac": "Export",
                "cost_interconnection_usd_per_mw": "POI",
                "cost_reinforcement_usd_per_mw_ac": "Reinforcement",
                "cost_total_trans_usd_per_mw_ac": "Total",
            }
            trans_dist_vars = {
                "dist_export_km": "Export",
                "dist_spur_km": "POI",
                "dist_reinforcement_km": "Reinforcement",
            }
        else:
            trans_cost_vars = {
                "cost_interconnection_usd_per_mw": "POI",
                "cost_reinforcement_usd_per_mw_ac": "Reinforcement",
                "cost_total_trans_usd_per_mw_ac": "Total",
            }
            trans_dist_vars = {
                "dist_spur_km": "POI",
                "dist_reinforcement_km": "Reinforcement",
            }

        trans_cost_df = scenario_df[list(trans_cost_vars.keys())].melt(
            value_name="cost_per_mw"
        )
        trans_cost_df["Transmission Component"] = trans_cost_df["variable"].replace(
            trans_cost_vars
        )
        trans_cost_df["Cost ($/MW)"] = trans_cost_df["cost_per_mw"] / 1e6
        trans_cost_df.replace(to_replace=np.inf, value=np.NAN, inplace=True)
        trans_cost_df.dropna(axis=0, inplace=True)

        # extract transmission distance data in tidy/long format
        trans_dist_df = scenario_df[list(trans_dist_vars.keys())].melt(
            value_name="Distance (km)"
        )
        trans_dist_df["Transmission Component"] = trans_dist_df["variable"].replace(
            trans_dist_vars
        )
        trans_dist_df.replace(to_replace=np.inf, value=np.NAN, inplace=True)
        trans_dist_df.dropna(axis=0, inplace=True)

        with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
            RC_FONT_PARAMS
        ):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2 * 6.5, 5))
            panel_1 = sns.boxplot(
                trans_cost_df,
                x="Transmission Component",
                y="Cost ($/MW)",
                showfliers=False,
                dodge=False,
                width=0.5,
                ax=ax[0],
                legend=False,
                color="#9ebcda",
            )
            panel_1 = format_graph(
                panel_1,
                xlabel=None,
                ylabel="Cost (million $/MW)",
                y_formatter=ticker.StrMethodFormatter("{x:,.1f}"),
            )

            panel_2 = sns.boxplot(
                trans_dist_df,
                x="Transmission Component",
                y="Distance (km)",
                showfliers=False,
                dodge=False,
                width=1 / 3,
                ax=ax[1],
                legend=False,
                color="#9ebcda",
            )
            panel_2 = format_graph(panel_2, xlabel=None, ylabel="Distance (km)")

            scenario_outname = scenario_name[0].replace(" ", "_").lower()
            out_image_path = out_path.joinpath(
                f"transmission_cost_dist_boxplot_{scenario_outname}.png"
            )
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=dpi, transparent=True)
            plt.close(fig)

    LOGGER.info("Plotting boxplots")
    boxplot_vars = {
        "lcoe": {
            "All-in-LCOE ($/MWh)": config.lcoe_all_in_col,
            "Site LCOE ($/MWh)": config.lcoe_site_col,
        },
        "trans_dist": {
            "Point-of-Interconnect Distance (km)": "dist_spur_km",
            "Reinforcement Distance (km)": "dist_reinforcement_km",
        },
        "trans_cost": {
            "Point-of-Interconnect Costs ($/MW)": "cost_interconnection_usd_per_mw",
            "Reinforcement Costs ($/MW)": "cost_reinforcement_usd_per_mw_ac",
        },
        "Project Site Capacity (MW)": "capacity_mw",
    }
    if config.tech in WIND:
        if "losses_wakes_pct" in all_df.columns:
            boxplot_vars["Wake Losses (%)"] = "losses_wakes_pct"
    if config.tech == "osw":
        # add plots for export cable costs and distance
        # weird syntax below is to ensure Export Cable plots are first
        boxplot_vars["trans_dist"] = {
            "Export Cable Distance (km)": "dist_export_km"
        } | boxplot_vars["trans_dist"]
        boxplot_vars["trans_cost"] = {
            "Export Cable Costs ($/MW)": "cost_export_usd_per_mw_ac"
        } | boxplot_vars["trans_cost"]

    for label, var_map in boxplot_vars.items():
        if isinstance(var_map, dict):
            out_filename = label
            n_panels = len(var_map)
            yvars = list(var_map.values())
            # get the maximum value to use on the y axis
            # use simple boxplot to get this
            ymax = 0
            ymin = 0
            for scenario_df in scenario_dfs:
                dummy_boxplot = scenario_df.boxplot(
                    column=yvars, return_type=None, showfliers=False
                )
                ymax = max(max(dummy_boxplot.get_ylim()) * 1.05, ymax)
                # pylint: disable-next=nested-min-max
                ymin = min(min(dummy_boxplot.get_ylim()), ymin)
                plt.close(dummy_boxplot.figure)
            with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
                RC_FONT_PARAMS
            ):
                fig, ax = plt.subplots(
                    nrows=1, ncols=n_panels, figsize=(n_panels * 6.5, 5)
                )

                for i, var_label in enumerate(var_map):
                    var = var_map[var_label]
                    panel = sns.boxplot(
                        all_df[~all_df[var].isna()],
                        x="Scenario",
                        y=var,
                        palette=scenario_palette,
                        showfliers=False,
                        dodge=False,
                        width=0.5,
                        ax=ax[i],
                        hue="Scenario",
                        legend=False,
                    )
                    panel = format_graph(
                        panel,
                        xlabel=None,
                        ylabel=var_label,
                        drop_legend=True,
                        ymax=ymax,
                        ymin=ymin,
                    )
                    wrap_labels(panel, 10)
                    panel.set_xticks(panel.get_xticks())
                    panel.set_xticklabels(panel.get_xticklabels(), weight="bold")

        elif isinstance(var_map, str):
            var = var_map
            var_label = label
            out_filename = (
                var_label.split(" (", maxsplit=1)[0].replace(" ", "_").lower()
            )
            ymax = 0
            ymin = 0
            for scenario_df in scenario_dfs:
                dummy_boxplot = scenario_df.boxplot(
                    column=var, return_type=None, showfliers=False
                )
                ymax = max(max(dummy_boxplot.get_ylim()) * 1.05, ymax)
                # pylint: disable-next=nested-min-max
                ymin = min(min(dummy_boxplot.get_ylim()), ymin)
                plt.close(dummy_boxplot.figure)
            with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
                RC_FONT_PARAMS
            ):
                fig, ax = plt.subplots(figsize=(6.5, 5))
                g = sns.boxplot(
                    all_df,
                    x="Scenario",
                    y=var,
                    palette=scenario_palette,
                    showfliers=False,
                    dodge=False,
                    width=0.5,
                    ax=ax,
                    hue="Scenario",
                    legend=False,
                )
                g = format_graph(
                    g,
                    xlabel=None,
                    ylabel=var_label,
                    drop_legend=True,
                    ymax=ymax,
                    ymin=ymin,
                )
                wrap_labels(g, 10)
        else:
            raise TypeError("Unexpected type: expected dict or str.")
        out_image_path = out_path.joinpath(f"{out_filename}_boxplots.png")
        plt.tight_layout()
        fig.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)

    LOGGER.info("Plotting histograms")
    hist_vars = [
        {
            "var": config.lcoe_all_in_col,
            "fmt_kwargs": {
                "ylabel": "Project Site Area (sq. km.)",
                "xmax": config.plots.total_lcoe_max,
                "xlabel": "All-In LCOE ($/MWh)",
                "xmin": 0,
            },
            "hist_kwargs": {
                "bins": 30,
                "weights": "area_developable_sq_km",
                "binrange": (0, config.plots.total_lcoe_max),
            },
        },
        {
            "var": "capacity_mw",
            "fmt_kwargs": {
                "ylabel": "Project Site Count",
                "xlabel": "Project Capacity (MW)",
            },
            "hist_kwargs": {"binwidth": 6} if config.tech in WIND else {},
        },
    ]
    if config.tech in WIND:
        if "n_turbines" in all_df.columns:
            n_turbines_max = (all_df["n_turbines"].max() + 5).round(-1)
            hist_vars.append(
                {
                    "var": "n_turbines",
                    "fmt_kwargs": {
                        "ylabel": "Project Site Count",
                        "xlabel": "Number of Turbines",
                    },
                    "hist_kwargs": {"binwidth": 5, "binrange": (0, n_turbines_max)},
                }
            )
        if "losses_wakes_pct" in all_df.columns:
            losses_wakes_pct_max = all_df["losses_wakes_pct"].max()
            hist_vars.append(
                {
                    "var": "losses_wakes_pct",
                    "fmt_kwargs": {
                        "ylabel": "Project Site Count",
                        "xlabel": "Wake Losses (%)",
                    },
                    "hist_kwargs": {
                        "binwidth": 0.5,
                        "binrange": (0, losses_wakes_pct_max),
                    },
                }
            )

    for hist_var in hist_vars:
        with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
            RC_FONT_PARAMS
        ):
            fig, ax = plt.subplots(figsize=(8, 5))

            xvar = hist_var["var"]
            g = sns.histplot(
                all_df,
                x=xvar,
                element="step",
                fill=False,
                hue="Scenario",
                palette=scenario_palette,
                ax=ax,
                **hist_var["hist_kwargs"],
            )
            g = format_graph(g, **hist_var["fmt_kwargs"])
            legend_lines = g.get_legend().get_lines()
            for i, line in enumerate(reversed(g.lines)):
                new_width = line.get_linewidth() + i * 0.75
                line.set_linewidth(new_width)
                legend_lines[i].set_linewidth(new_width)
            out_image_path = out_path.joinpath(f"{xvar}_histogram.png")
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=dpi, transparent=True)
            plt.close(fig)

    LOGGER.info("Plotting regional boxplots")
    regbox_vars = [
        {
            "var": config.lcoe_all_in_col,
            "fmt_kwargs": {"xlabel": "All-in LCOE ($/MWh)"},
            "box_kwargs": {},
        },
        {
            "var": "lcot_usd_per_mwh",
            "fmt_kwargs": {"xlabel": "Levelized Cost of Transmission ($/MWh)"},
            "box_kwargs": {},
        },
        {
            "var": "capacity_density",
            "fmt_kwargs": {"xlabel": "Capacity Density (MW/sq. km.)"},
            "box_kwargs": {},
        },
    ]
    ordered_regions = ORDERED_REGIONS
    if config.tech == "osw":
        ordered_regions = list(
            all_df.groupby("offtake_state")
            .sum("capacity_ac_mw")
            .sort_values("capacity_ac_mw", ascending=False)
            .index
        )
    for regbox_var in regbox_vars:
        xvar = regbox_var["var"]
        with sns.axes_style("whitegrid", DEFAULT_RC_PARAMS), plt.rc_context(
            RC_FONT_PARAMS
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            g = sns.boxplot(
                all_df.reset_index(),
                x=xvar,
                y=region_col,
                hue="Scenario",
                showfliers=False,
                order=ordered_regions,
                palette=scenario_palette,
                legend=True,
                dodge=True,
                gap=0.3,
                ax=ax,
                **regbox_var["box_kwargs"],
            )
            g = format_graph(
                g, ylabel="Region", move_legend_outside=True, **regbox_var["fmt_kwargs"]
            )
            if config.tech == "osw":
                g.set_yticks(g.get_yticks())
                g.set_yticklabels(g.get_yticklabels(), fontsize=10)
            out_image_path = out_path.joinpath(f"{xvar}_regional_boxplots.png")
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=dpi, transparent=True)
            plt.close(fig)

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
    """
    Create standard set of report maps for input supply curves.
    """
    # pylint: disable=too-many-statements, too-many-branches
    LOGGER.info("Starting plot creation")

    LOGGER.info(f"Loading configuration file {config_file}")
    try:
        config = Config.from_json(config_file)
    except ValidationError:
        LOGGER.exception("Input configuration file failed. Exiting process.")
        sys.exit(1)
    LOGGER.info("Configuration file loaded.")

    n_scenarios = len(config.scenarios)
    if n_scenarios > 4:
        err = "Cannot plot more than 4 scenarios."
        LOGGER.error(err)
        sys.exit(1)
    LOGGER.info(f"{n_scenarios} supply curve scenarios will be plotted:")
    for scenario in config.scenarios:
        LOGGER.info(f"\t{scenario.name}: {scenario.source.name}")

    # make output directory (only if needed)
    out_path.mkdir(parents=False, exist_ok=True)

    LOGGER.info("Loading boundaries dataset")
    boundaries_gdf = gpd.read_file(DEFAULT_BOUNDARIES)
    boundaries_gdf.to_crs("EPSG:4326", inplace=True)
    boundaries_singlepart_gdf = boundaries_gdf.explode(index_parts=True)
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

    if config.cf_col is None:
        cf_col = "capacity_factor_ac"
    else:
        cf_col = config.cf_col

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
                    "breaks": [500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000],
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
        raise ValueError(
            f"Invalid input: tech={config.tech}. Valid options are: ['wind', 'pv']."
        )

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
    if n_scenarios > 4:
        err = "Cannot map more than 4 scenarios."
        LOGGER.error(err)
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
                        central_longitude=center_lon, central_latitude=center_lat
                    )
                },
            )
            for i, scenario_name in enumerate(scenario_dfs):
                scenario_df = scenario_dfs[scenario_name]
                if map_var not in scenario_df.columns:
                    err = (
                        f"{map_var} column not found in one or more input supply "
                        "curves. Consider using the `exclude_maps` configuration "
                        "option to skip map generation for this column."
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
                    boundaries_df=boundaries_singlepart_gdf,
                    extent=map_extent,
                    layer_kwargs={"s": point_size, "linewidth": 0, "marker": "o"},
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
            if n_scenarios in (3, 4):
                panel_width = 0.6
                panel_height = 0.52
                panel_dims = [panel_width, panel_height]

                lower_lefts = [
                    [mid_xcoord, min_ycoord],
                    [min_xcoord, min_ycoord],
                    [mid_xcoord, mid_ycoord],
                    [min_xcoord, mid_ycoord],
                ]
                for j in range(0, n_panels):
                    coords = lower_lefts[j]
                    ax.ravel()[-j - 1].set_position(coords + panel_dims)
            elif n_scenarios in (1, 2):
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
                if n_rows == 2:
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

            lpanel = fig.add_subplot(alpha=0, frame_on=False)
            lpanel.set_axis_off()
            lpanel.set_position(legend_panel_position)

            lpanel.legend(
                legend_handles,
                legend_texts,
                frameon=False,
                loc=legend_loc,
                title=map_settings["legend_title"],
                ncol=legend_cols,
                handletextpad=-0.1,
                columnspacing=0,
                fontsize=legend_font_size,
                title_fontproperties={"size": legend_font_size, "weight": "bold"},
            )

            out_image_name = f"{map_var}.png"
            out_image_path = out_path.joinpath(out_image_name)
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
    help=("Cell size in meters of characterization layers. " "Default is 90."),
)
def unpack_characterizations(supply_curve_csv, char_map, out_csv, cell_size=90.0):
    """
    Unpacks characterization data from the input supply curve dataframe,
    converting values from embedded JSON strings to new standalone columns,
    and saves out a new version of the supply curve with these columns
    included.
    """

    LOGGER.info("Loading supply curve data")
    supply_curve_df = pd.read_csv(supply_curve_csv)

    LOGGER.info("Loading characterization mapping")
    with open(char_map, "r") as f:
        characterization_map = json.load(f)

    LOGGER.info("Unpacking characterizations")
    char_df = characterizations.unpack_characterizations(
        supply_curve_df, characterization_map, cell_size
    )

    char_df.to_csv(out_csv, header=True, index=False)
    LOGGER.info("Command completed successfully.")
