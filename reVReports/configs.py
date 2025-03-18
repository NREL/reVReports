# -*- coding: utf-8 -*-
"""Configuration module"""
# pylint: disable=too-few-public-methods, disable=no-self-argument
from typing import List
import json
from pathlib import Path
from pydantic import BaseModel, field_validator
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

DEFAULT_COLORS = [to_hex(rgb) for rgb in plt.colormaps["tab10"].colors]


class BaseModelStrict(BaseModel):
    """
    Customizing BaseModel to perform strict checking that will raise a ValidationError
    for extra parameters.
    """

    model_config = {"extra": "forbid"}


class SupplyCurveScenario(BaseModelStrict):
    """
    Inputs for an indiviual supply curve scenario.
    """

    source: Path
    name: str
    color: str = None

    @field_validator("source")
    def expand_user(cls, value):
        """
        Expand user directory of input source paths

        Parameters
        ----------
        value : pathlib.Path
            Input source path

        Returns
        -------
        pathlib.Path
            Source path with user directory expanded, if applicable
        """
        return value.expanduser()


class Plots(BaseModelStrict):
    """
    Container for settings related to specific plots.
    """

    site_lcoe_max: float = 70
    total_lcoe_max: float = 100


class MapVar(BaseModelStrict):
    """
    Container for settings related to variables that will be mapped
    """

    column: str
    breaks: List[float]
    cmap: str
    legend_title: str


VALID_TECHS = ["wind", "osw", "pv", "geo"]


class Config(BaseModelStrict):
    """
    Configuration settings for creating plots.
    """

    tech: str
    scenarios: List[SupplyCurveScenario] = []
    plots: Plots = Plots()
    map_vars: List[MapVar] = []
    exclude_maps: List[str] = []
    lcoe_site_col: str = "lcoe_site_usd_per_mwh"
    lcoe_all_in_col: str = "lcoe_all_in_usd_per_mwh"
    cf_col: str = None

    @field_validator("scenarios")
    def default_scenario_colors(cls, value):
        """
        If input scenarios do not have a color set, this function sets them to values
        from the tab10 colormap. This is handled at the Config level rather than the
        SupplyCurveScenario level so that the colormap can be incremented for each
        scenario.

        Parameters
        ----------
        value : List[SupplyCurveScenario]
            List of SupplyCurveScenario models.

        Returns
        -------
         List[SupplyCurveScenario]
            List of SupplyCurveScenarios with default colors set, if needed.
        """
        for i, scenario in enumerate(value):
            if scenario.color is None:
                scenario.color = DEFAULT_COLORS[i]

        return value

    @field_validator("tech")
    def valid_tech(cls, value):
        """
        Check that the input value for tech is one of the valid options.

        Parameters
        ----------
        value : str
            Input value for 'tech'

        Returns
        -------
        str
            Returns the input value (as long as it is one of the valid options)

        Raises
        ------
        ValueError
            A ValueError will be raised if the input value is not a valid option.
        """
        if value not in VALID_TECHS:
            raise ValueError(
                f"Input tech '{value}' is invalid. Valid options are: {VALID_TECHS}"
            )

        return value

    @classmethod
    def from_json(cls, json_path: Path):
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        json_path : [pathlib.Path, str]
            Path to JSON file containing input settings.

        Returns
        -------
        Config
            Configuration settings.
        """
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return cls(**json_data)
