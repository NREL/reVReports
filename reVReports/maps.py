"""reVReports map generation functions"""

import logging
from functools import cached_property
import warnings

import pandas as pd
import numpy as np
import tqdm
import geoplot as gplt
import geopandas as gpd
from matplotlib import pyplot as plt

from reVReports.configs import VALID_TECHS
from reVReports.utilities.maps import (
    map_geodataframe_column,
    DEFAULT_BOUNDARIES,
)
from reVReports.utilities.plots import SMALL_SIZE, BIGGER_SIZE, RC_FONT_PARAMS
from reVReports.exceptions import reVReportsValueError

logger = logging.getLogger(__name__)


class _BoundariesData:
    @cached_property
    def _boundaries_gdf_raw(self):
        return gpd.read_file(DEFAULT_BOUNDARIES).to_crs("EPSG:4326")

    @cached_property
    def _boundaries_dissolved(self):
        return self._boundaries_gdf_raw.union_all()

    @cached_property
    def background_gdf(self):
        return gpd.GeoDataFrame(
            {"geometry": [self._boundaries_dissolved]},
            crs=self._boundaries_gdf_raw.crs,
        ).explode(index_parts=False)

    @cached_property
    def boundaries_single_part_gdf(self):
        return self._boundaries_gdf_raw.explode(index_parts=True)

    @cached_property
    def map_extent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.background_gdf.buffer(0.01).total_bounds

    @cached_property
    def center_lon(self):
        return self._boundaries_dissolved.centroid.x

    @cached_property
    def center_lat(self):
        return self._boundaries_dissolved.centroid.y


_BOUNDARIES = _BoundariesData()


class MapData:
    def __init__(self, config, cap_col):
        self._config = config
        self.cap_col = cap_col

    def __iter__(self):
        return iter(self.scenario_dfs.items())

    @property
    def config(self):
        return self._config

    @cached_property
    def scenario_dfs(self):
        logger.info("Loading and augmenting supply curve data")
        scenario_dfs = {}
        for scenario in tqdm.tqdm(
            self._config.scenarios, total=len(self._config.scenarios)
        ):
            scenario_df = pd.read_csv(scenario.source)

            # drop zero capacity sites
            scenario_sub_df = scenario_df[
                scenario_df["capacity_ac_mw"] > 0
            ].copy()

            supply_curve_gdf = gpd.GeoDataFrame(
                scenario_sub_df,
                geometry=gpd.points_from_xy(
                    x=scenario_sub_df["longitude"],
                    y=scenario_sub_df["latitude"],
                ),
                crs="EPSG:4326",
            )
            supply_curve_gdf["capacity_density"] = (
                supply_curve_gdf[self.cap_col]
                / supply_curve_gdf["area_developable_sq_km"].replace(0, np.nan)
            ).replace(np.nan, 0)

            scenario_dfs[scenario.name] = supply_curve_gdf

        return scenario_dfs


class MapGenerator:
    def __init__(self, map_data):
        self._map_data = map_data

    @property
    def num_scenarios(self):
        return len(self._map_data.scenario_dfs)

    def build_maps(self, map_vars, out_directory, dpi, point_size=2.0):
        n_cols = 2
        n_rows = int(np.ceil(self.num_scenarios / n_cols))
        logger.info("Creating maps")
        for map_var, map_settings in tqdm.tqdm(map_vars.items()):
            with plt.rc_context(RC_FONT_PARAMS):
                fig, ax = plt.subplots(
                    ncols=n_cols,
                    nrows=n_rows,
                    figsize=(13, 4 * n_rows),
                    subplot_kw={
                        "projection": gplt.crs.AlbersEqualArea(
                            central_longitude=_BOUNDARIES.center_lon,
                            central_latitude=_BOUNDARIES.center_lat,
                        )
                    },
                )
                for i, (scenario_name, scenario_df) in enumerate(
                    self._map_data
                ):
                    # scenario_df = scenario_dfs[scenario_name]
                    if map_var not in scenario_df.columns:
                        err = (
                            f"{map_var} column not found in one or more input "
                            "supply curves. Consider using the `exclude_maps` "
                            "configuration option to skip map generation for "
                            "this column."
                        )
                        logger.error(err)
                    panel = ax.ravel()[i]
                    panel = map_geodataframe_column(
                        scenario_df,
                        map_var,
                        color_map=map_settings.get("cmap"),
                        breaks=map_settings.get("breaks"),
                        map_title=None,
                        legend_title=map_settings.get("legend_title"),
                        background_df=_BOUNDARIES.background_gdf,
                        boundaries_df=_BOUNDARIES.boundaries_single_part_gdf,
                        extent=_BOUNDARIES.map_extent,
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
                        legend=(i + 1 == self.num_scenarios),
                        ax=panel,
                    )
                    panel.patch.set_alpha(0)
                    panel.set_title(scenario_name, y=0.88)

                n_panels = len(ax.ravel())
                min_xcoord = -0.04
                mid_xcoord = 0.465
                min_ycoord = 0.0
                mid_ycoord = 0.475
                if self.num_scenarios in {3, 4}:
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
                elif self.num_scenarios in {1, 2}:
                    ax.ravel()[0].set_position([-0.25, 0.0, 1, 1])
                    ax.ravel()[1].set_position([0.27, 0.0, 1, 1])

                if self.num_scenarios < n_panels:
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
                out_image_path = out_directory / out_image_name
                fig.savefig(out_image_path, dpi=dpi, transparent=True)
                plt.close(fig)


def configure_map_params(config):
    logger.info("Configuring map settings")
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

    return cap_col, point_size, map_vars
