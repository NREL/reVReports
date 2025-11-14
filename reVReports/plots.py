"""reVReports plot generating functions"""

import logging
from functools import cached_property

import pandas as pd
import numpy as np
import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

from reVReports.data import augment_sc_df, ORDERED_REGIONS
from reVReports.utilities.plots import (
    format_graph,
    wrap_labels,
    DEFAULT_RC_PARAMS,
    NO_OUTLINE_RC_PARAMS,
    SMALL_SIZE,
    SMALL_MEDIUM_SIZE,
    RC_FONT_PARAMS,
)
from reVReports.exceptions import reVReportsTypeError


WIND = {"wind", "osw"}
logger = logging.getLogger(__name__)


class PlotData:
    """Load and organize supply curve inputs for plotting

    Parameters
    ----------
    config : object
        Plotting configuration with scenario definitions and column
        names.
    """

    def __init__(self, config):
        """Store configuration and defer data loading until needed

        Parameters
        ----------
        config : object
            Plotting configuration with scenario definitions and column
            names.
        """
        self._config = config
        self._all_df = None
        self._scenario_dfs = None

    @property
    def config(self):
        """Configuration object for the current plot generation

        Returns
        -------
        object
            Plotting configuration with scenario definitions and column
            names.
        """
        return self._config

    @property
    def all_df(self):
        """Combined dataframe of all scenarios with augmented fields

        Returns
        -------
        pandas.DataFrame
            Aggregated and augmented supply curve records for every
            scenario.
        """
        if self._all_df is None:
            self._load_and_augment_supply_curve_data()
        return self._all_df

    @property
    def scenario_dfs(self):
        """List of per-scenario dataframes with augmented fields

        Returns
        -------
        list of pandas.DataFrame
            Each entry contains augmented supply curve data for one
            scenario.
        """
        if self._scenario_dfs is None:
            self._load_and_augment_supply_curve_data()
        return self._scenario_dfs

    @cached_property
    def top_level_sums_df(self):
        """Aggregate top level metrics across scenarios

        Returns
        -------
        pandas.DataFrame
            Scenario level totals with capacity and annual energy sums.
        """
        top_level_sums_df = (
            self.all_df.groupby("Scenario")[
                [
                    "area_developable_sq_km",
                    "capacity_mw",
                    "annual_energy_site_mwh",
                ]
            ]
            .sum()
            .reset_index()
        )
        top_level_sums_df["capacity_gw"] = (
            top_level_sums_df["capacity_mw"] / 1000.0
        )
        top_level_sums_df["aep_twh"] = (
            top_level_sums_df["annual_energy_site_mwh"] / 1000.0 / 1000
        )
        return top_level_sums_df

    def _load_and_augment_supply_curve_data(self):
        logger.info("Loading and augmenting supply curve data")
        self._scenario_dfs = []
        for i, scenario in tqdm.tqdm(
            enumerate(self._config.scenarios),
            total=len(self._config.scenarios),
        ):
            scenario_df = pd.read_csv(scenario.source)

            try:
                aug_df = augment_sc_df(
                    scenario_df,
                    scenario_name=scenario.name,
                    scenario_number=i,
                    tech=self._config.tech,
                    lcoe_all_in_col=self._config.lcoe_all_in_col,
                )
            except KeyError:
                logger.warning(
                    "Required columns are missing from the input supply "
                    "curve. Was your supply curve created by reV≥v0.14.5?"
                )
                raise

            # drop sites with zero capacity
            # (this also removes inf values for total_lcoe)
            aug_df_w_capacity = aug_df[aug_df["capacity_mw"] > 0].copy()

            self._scenario_dfs.append(aug_df_w_capacity)

        # combine the data into a single data frame
        self._all_df = pd.concat(self._scenario_dfs).sort_values(
            by=["scenario_number", self._config.lcoe_all_in_col],
            ascending=True,
        )


class PlotGenerator:
    """Build plots from prepared supply curve dataframes

    Parameters
    ----------
    plot_data : PlotData
        Data interface that exposes scenario and combined dataframes.
    out_directory : pathlib.Path
        Directory where generated plot images are written.
    dpi : int
        Resolution used when saving matplotlib figures.
    """

    def __init__(self, plot_data, out_directory, dpi):
        """Assign dependencies for plotting routines

        Parameters
        ----------
        plot_data : PlotData
            Data interface that exposes scenario and combined
            dataframes.
        out_directory : pathlib.Path
            Directory where generated plot images are written.
        dpi : int
            Resolution used when saving matplotlib figures.
        """
        self._plot_data = plot_data
        self._config = plot_data.config
        self.out_directory = out_directory
        self.dpi = dpi

    @property
    def all_df(self):
        """Combined dataframe with augmented supply curve metrics

        Returns
        -------
        pandas.DataFrame
            Aggregated and augmented supply curve records for every
            scenario.
        """
        return self._plot_data.all_df

    @property
    def scenario_dfs(self):
        """Per scenario dataframes with augmented supply curve metrics

        Returns
        -------
        list of pandas.DataFrame
            Each entry contains augmented supply curve data for one
            scenario.
        """
        return self._plot_data.scenario_dfs

    def build_supply_curves(self):
        """Create supply curve line plots for capacity and generation"""
        logger.info("Plotting supply curves")
        # Prepare data for plotting supply curves
        # Set up data frame we can use to plot all-in
        # and site lcoe together on supply curve plots
        supply_curve_total_lcoe_df = self.all_df[
            [
                self._config.lcoe_all_in_col,
                "capacity_mw",
                "annual_energy_site_mwh",
                "cumul_capacit_gw",
                "cumul_aep_twh",
                "Scenario",
            ]
        ].copy()
        supply_curve_total_lcoe_df["LCOE Value"] = "All-In LCOE"
        supply_curve_total_lcoe_df = supply_curve_total_lcoe_df.rename(
            columns={self._config.lcoe_all_in_col: "lcoe"}
        )

        supply_curve_site_lcoe_df = self.all_df[
            [
                self._config.lcoe_site_col,
                "capacity_mw",
                "annual_energy_site_mwh",
                "Scenario",
            ]
        ].copy()
        supply_curve_site_lcoe_df = supply_curve_site_lcoe_df.sort_values(
            by=[self._config.lcoe_site_col], ascending=True
        )
        supply_curve_site_lcoe_df["cumul_capacit_gw"] = (
            supply_curve_site_lcoe_df.groupby("Scenario")[
                "capacity_mw"
            ].cumsum()
            / 1000
        )
        supply_curve_site_lcoe_df["cumul_aep_twh"] = (
            supply_curve_site_lcoe_df.groupby("Scenario")[
                "annual_energy_site_mwh"
            ].cumsum()
            / 1000
            / 1000
        )
        supply_curve_site_lcoe_df["LCOE Value"] = "Site LCOE"
        supply_curve_site_lcoe_df = supply_curve_site_lcoe_df.rename(
            columns={self._config.lcoe_site_col: "lcoe"}
        )

        supply_curve_df = pd.concat(
            [supply_curve_total_lcoe_df, supply_curve_site_lcoe_df]
        )

        # Supply curves - two panel figure showing cumulative capacity
        # and generation by LCOE
        if self._config.lcoe_all_in_col != self._config.lcoe_site_col:
            sc_line_style = "LCOE Value"
        else:
            sc_line_style = None
        with (
            sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
            plt.rc_context(
                RC_FONT_PARAMS
                | {
                    "xtick.labelsize": SMALL_SIZE,
                    "ytick.labelsize": SMALL_SIZE,
                }
            ),
        ):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
            panel_1 = sns.lineplot(
                data=supply_curve_df,
                y="lcoe",
                x="cumul_capacit_gw",
                hue="Scenario",
                palette=self._config.scenario_palette,
                hue_order=self._config.scenario_palette,
                style=sc_line_style,
                ax=ax[0],
            )
            panel_1 = format_graph(
                panel_1,
                xmin=0,
                xmax=None,
                ymin=0,
                ymax=self._config.plots.site_lcoe_max,
                xlabel="Cumulative Capacity (GW)",
                ylabel="Levelized Cost of Energy ($/MWh)",
                drop_legend=True,
            )
            panel_2 = sns.lineplot(
                data=supply_curve_df,
                y="lcoe",
                x="cumul_aep_twh",
                hue="Scenario",
                palette=self._config.scenario_palette,
                hue_order=self._config.scenario_palette,
                style=sc_line_style,
                ax=ax[1],
            )
            panel_2 = format_graph(
                panel_2,
                xmin=0,
                xmax=None,
                ymin=0,
                ymax=self._config.plots.site_lcoe_max,
                xlabel="Cumulative Annual Generation (TWh)",
                ylabel="Levelized Cost of Energy ($/MWh)",
                move_legend_outside=True,
            )
            out_image_path = self.out_directory / "supply_curves.png"
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
            plt.close(fig)

        # single panel supply curve - LCOE vs cumulative capacity
        with (
            sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
            plt.rc_context(RC_FONT_PARAMS),
        ):
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax = sns.lineplot(
                data=supply_curve_df,
                y="lcoe",
                x="cumul_capacit_gw",
                hue="Scenario",
                palette=self._config.scenario_palette,
                hue_order=self._config.scenario_palette,
                style=sc_line_style,
                ax=ax,
            )
            # handles, labels = ax.get_legend_handles_labels()
            ax = format_graph(
                ax,
                xmin=0,
                xmax=None,
                ymin=0,
                ymax=self._config.plots.site_lcoe_max,
                xlabel="Cumulative Capacity (GW)",
                ylabel="Levelized Cost of Energy ($/MWh)",
                drop_legend=False,
            )
            sns.move_legend(
                ax, "lower center", ncol=2, fontsize=SMALL_MEDIUM_SIZE
            )
            out_image_path = (
                self.out_directory / "supply_curves_capacity_only.png"
            )
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
            plt.close(fig)

    def build_capacity_by_region_bar_chart(self):
        """Create bar chart of economic capacity by region"""
        logger.info("Plotting capacity by region and scenario barchart")
        # Regional capacity comparison
        # Sum the capacity by nrel region
        region_col = (
            "offtake_state" if self._config.tech == "osw" else "nrel_region"
        )
        econ_cap_by_region_df = (
            self.all_df[
                self.all_df[self._config.lcoe_all_in_col]
                <= self._config.plots.total_lcoe_max
            ]
            .groupby(["Scenario", region_col])["capacity_mw"]
            .sum()
            .reset_index()
        )
        econ_cap_by_region_df["capacity_gw"] = (
            econ_cap_by_region_df["capacity_mw"] / 1000
        )
        econ_cap_by_region_df = econ_cap_by_region_df.sort_values(
            "capacity_gw", ascending=False
        )

        with (
            sns.axes_style("whitegrid", NO_OUTLINE_RC_PARAMS),
            plt.rc_context(RC_FONT_PARAMS),
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            g = sns.barplot(
                econ_cap_by_region_df,
                y=region_col,
                x="capacity_gw",
                hue="Scenario",
                dodge=True,
                palette=self._config.scenario_palette,
                ax=ax,
            )
            g = format_graph(g, xlabel="Total Capacity (GW)", ylabel="Region")
            if self._config.tech == "osw":
                g.set_yticks(g.get_yticks())
                g.set_yticklabels(g.get_yticklabels(), fontsize=10)
            out_image_path = (
                self.out_directory / "regional_capacity_barchart.png"
            )
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
            plt.close(fig)

    def build_transmission_box_plots(self):
        """Create box plots of transmission costs and distances"""
        logger.info("Plotting Transmission Cost and Distance Box plots")
        for scenario_name, scenario_df in self.all_df.groupby(
            ["Scenario"], as_index=False
        ):
            # extract transmission cost data in tidy/long format
            if self._config.tech == "osw":
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
            trans_cost_df["Transmission Component"] = trans_cost_df[
                "variable"
            ].replace(trans_cost_vars)
            trans_cost_df["Cost ($/MW)"] = trans_cost_df["cost_per_mw"] / 1e6
            trans_cost_df = trans_cost_df.replace(
                to_replace=np.inf, value=np.nan
            )
            trans_cost_df = trans_cost_df.dropna(axis=0)

            # extract transmission distance data in tidy/long format
            trans_dist_df = scenario_df[list(trans_dist_vars.keys())].melt(
                value_name="Distance (km)"
            )
            trans_dist_df["Transmission Component"] = trans_dist_df[
                "variable"
            ].replace(trans_dist_vars)
            trans_dist_df.replace(to_replace=np.inf, value=np.nan)
            trans_dist_df = trans_dist_df.replace(
                to_replace=np.inf, value=np.nan
            )
            trans_dist_df = trans_dist_df.dropna(axis=0)

            with (
                sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
                plt.rc_context(RC_FONT_PARAMS),
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
                panel_2 = format_graph(
                    panel_2, xlabel=None, ylabel="Distance (km)"
                )

                scenario_outname = scenario_name[0].replace(" ", "_").lower()
                out_image_path = (
                    self.out_directory
                    / f"transmission_cost_dist_boxplot_{scenario_outname}.png"
                )
                plt.tight_layout()
                fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
                plt.close(fig)

    def build_box_plots(self):
        """Create box plots for scenario level metrics"""
        logger.info("Plotting box plots")
        boxplot_vars = {
            "lcoe": {
                "All-in-LCOE ($/MWh)": self._config.lcoe_all_in_col,
                "Site LCOE ($/MWh)": self._config.lcoe_site_col,
            },
            "trans_dist": {
                "Point-of-Interconnect Distance (km)": "dist_spur_km",
                "Reinforcement Distance (km)": "dist_reinforcement_km",
            },
            "trans_cost": {
                "Point-of-Interconnect Costs ($/MW)": (
                    "cost_interconnection_usd_per_mw"
                ),
                "Reinforcement Costs ($/MW)": (
                    "cost_reinforcement_usd_per_mw_ac"
                ),
            },
            "Project Site Capacity (MW)": "capacity_mw",
        }
        if (
            self._config.tech in WIND
            and "losses_wakes_pct" in self.all_df.columns
        ):
            boxplot_vars["Wake Losses (%)"] = "losses_wakes_pct"
        if self._config.tech == "osw":
            # add plots for export cable costs and distance weird syntax
            # below is to ensure Export Cable plots are first
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
                y_vars = list(var_map.values())
                # get the maximum value to use on the y axis
                # use simple boxplot to get this
                ymax = 0
                ymin = 0
                for scenario_df in self.scenario_dfs:
                    dummy_boxplot = scenario_df.boxplot(
                        column=y_vars, return_type=None, showfliers=False
                    )
                    ymax = max(max(dummy_boxplot.get_ylim()) * 1.05, ymax)
                    ymin = min(min(dummy_boxplot.get_ylim()) * 1, ymin)
                    plt.close(dummy_boxplot.figure)
                with (
                    sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
                    plt.rc_context(RC_FONT_PARAMS),
                ):
                    fig, ax = plt.subplots(
                        nrows=1, ncols=n_panels, figsize=(n_panels * 6.5, 5)
                    )

                    for i, var_label in enumerate(var_map):
                        var = var_map[var_label]
                        panel = sns.boxplot(
                            self.all_df[~self.all_df[var].isna()],
                            x="Scenario",
                            y=var,
                            palette=self._config.scenario_palette,
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
                        panel.set_xticklabels(
                            panel.get_xticklabels(), weight="bold"
                        )

            elif isinstance(var_map, str):
                var = var_map
                var_label = label
                out_filename = (
                    var_label.split(" (", maxsplit=1)[0]
                    .replace(" ", "_")
                    .lower()
                )
                ymax = 0
                ymin = 0
                for scenario_df in self.scenario_dfs:
                    dummy_boxplot = scenario_df.boxplot(
                        column=var, return_type=None, showfliers=False
                    )
                    ymax = max(max(dummy_boxplot.get_ylim()) * 1.05, ymax)
                    ymin = min(min(dummy_boxplot.get_ylim()) * 1, ymin)
                    plt.close(dummy_boxplot.figure)
                with (
                    sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
                    plt.rc_context(RC_FONT_PARAMS),
                ):
                    fig, ax = plt.subplots(figsize=(6.5, 5))
                    g = sns.boxplot(
                        self.all_df,
                        x="Scenario",
                        y=var,
                        palette=self._config.scenario_palette,
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
                msg = "Unexpected type: expected dict or str"
                raise reVReportsTypeError(msg)
            out_image_path = (
                self.out_directory / f"{out_filename}_boxplots.png"
            )
            plt.tight_layout()
            fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
            plt.close(fig)

    def build_histograms(self):
        """Create histograms for core supply curve variables"""
        logger.info("Plotting histograms")
        hist_vars = [
            {
                "var": self._config.lcoe_all_in_col,
                "fmt_kwargs": {
                    "ylabel": "Project Site Area (sq. km.)",
                    "xmax": self._config.plots.total_lcoe_max,
                    "xlabel": "All-In LCOE ($/MWh)",
                    "xmin": 0,
                },
                "hist_kwargs": {
                    "bins": 30,
                    "weights": "area_developable_sq_km",
                    "binrange": (0, self._config.plots.total_lcoe_max),
                },
            },
            {
                "var": "capacity_mw",
                "fmt_kwargs": {
                    "ylabel": "Project Site Count",
                    "xlabel": "Project Capacity (MW)",
                },
                "hist_kwargs": {"binwidth": 6}
                if self._config.tech in WIND
                else {},
            },
        ]
        if self._config.tech in WIND:
            if "n_turbines" in self.all_df.columns:
                n_turbines_max = (self.all_df["n_turbines"].max() + 5).round(
                    -1
                )
                hist_vars.append(
                    {
                        "var": "n_turbines",
                        "fmt_kwargs": {
                            "ylabel": "Project Site Count",
                            "xlabel": "Number of Turbines",
                        },
                        "hist_kwargs": {
                            "binwidth": 5,
                            "binrange": (0, n_turbines_max),
                        },
                    }
                )
            if "losses_wakes_pct" in self.all_df.columns:
                losses_wakes_pct_max = self.all_df["losses_wakes_pct"].max()
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
            with (
                sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
                plt.rc_context(RC_FONT_PARAMS),
            ):
                fig, ax = plt.subplots(figsize=(8, 5))

                x_var = hist_var["var"]
                g = sns.histplot(
                    self.all_df,
                    x=x_var,
                    element="step",
                    fill=False,
                    hue="Scenario",
                    palette=self._config.scenario_palette,
                    ax=ax,
                    **hist_var["hist_kwargs"],
                )
                g = format_graph(g, **hist_var["fmt_kwargs"])
                legend_lines = g.get_legend().get_lines()
                for i, line in enumerate(reversed(g.lines)):
                    new_width = line.get_linewidth() + i * 0.75
                    line.set_linewidth(new_width)
                    legend_lines[i].set_linewidth(new_width)
                out_image_path = self.out_directory / f"{x_var}_histogram.png"
                plt.tight_layout()
                fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
                plt.close(fig)

    def build_regional_box_plots(self):
        """Create regional box plots for key metrics"""
        logger.info("Plotting regional box plots")
        region_col = (
            "offtake_state" if self._config.tech == "osw" else "nrel_region"
        )
        reg_box_vars = [
            {
                "var": self._config.lcoe_all_in_col,
                "fmt_kwargs": {"xlabel": "All-in LCOE ($/MWh)"},
                "box_kwargs": {},
            },
            {
                "var": "lcot_usd_per_mwh",
                "fmt_kwargs": {
                    "xlabel": "Levelized Cost of Transmission ($/MWh)"
                },
                "box_kwargs": {},
            },
            {
                "var": "capacity_density",
                "fmt_kwargs": {"xlabel": "Capacity Density (MW/sq. km.)"},
                "box_kwargs": {},
            },
        ]
        ordered_regions = ORDERED_REGIONS
        if self._config.tech == "osw":
            ordered_regions = list(
                self.all_df.groupby("offtake_state")
                .sum("capacity_ac_mw")
                .sort_values("capacity_ac_mw", ascending=False)
                .index
            )
        for reg_box_var in reg_box_vars:
            x_var = reg_box_var["var"]
            with (
                sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
                plt.rc_context(RC_FONT_PARAMS),
            ):
                fig, ax = plt.subplots(figsize=(8, 5))
                g = sns.boxplot(
                    self.all_df.reset_index(),
                    x=x_var,
                    y=region_col,
                    hue="Scenario",
                    showfliers=False,
                    order=ordered_regions,
                    palette=self._config.scenario_palette,
                    legend=True,
                    dodge=True,
                    gap=0.3,
                    ax=ax,
                    **reg_box_var["box_kwargs"],
                )
                g = format_graph(
                    g,
                    ylabel="Region",
                    move_legend_outside=True,
                    **reg_box_var["fmt_kwargs"],
                )
                if self._config.tech == "osw":
                    g.set_yticks(g.get_yticks())
                    g.set_yticklabels(g.get_yticklabels(), fontsize=10)

                out_image_path = (
                    self.out_directory / f"{x_var}_regional_boxplots.png"
                )
                plt.tight_layout()
                fig.savefig(out_image_path, dpi=self.dpi, transparent=True)
                plt.close(fig)


def make_bar_plot(
    data_df, y_col, ylabel, scenario_palette, out_image_path, dpi
):
    """Create a bar plot comparing scenario totals

    Parameters
    ----------
    data_df : pandas.DataFrame
        Dataframe containing pre-aggregated scenario totals.
    y_col : str
        Column name that holds the metric to plot on the y-axis.
    ylabel : str
        Axis label describing the plotted metric.
    scenario_palette : dict
        Mapping of scenario names to palette colors, used for ordering.
    out_image_path : pathlib.Path
        Destination path for the saved figure.
    dpi : int
        Resolution used when writing the figure to disk.
    """
    logger.info("Plotting total %s by scenario bar chart", ylabel)
    with (
        sns.axes_style("whitegrid", DEFAULT_RC_PARAMS),
        plt.rc_context(RC_FONT_PARAMS),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ymax = data_df[y_col].max() * 1.05
        g = sns.barplot(
            data_df,
            x="Scenario",
            y=y_col,
            hue="Scenario",
            palette=scenario_palette,
            order=scenario_palette,
            ax=ax,
            legend=False,
        )
        g = format_graph(g, xlabel=None, ylabel=ylabel, ymax=ymax)
        wrap_labels(g, 10)
        g.set_xticks(g.get_xticks())
        g.set_xticklabels(g.get_xticklabels(), weight="bold")
        plt.tight_layout()
        g.figure.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(fig)
