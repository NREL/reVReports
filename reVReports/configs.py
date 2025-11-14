"""Configuration module"""

import json
from pathlib import Path
from functools import cached_property

from pydantic import BaseModel, field_validator
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from reVReports.exceptions import reVReportsValueError


DEFAULT_COLORS = [to_hex(rgb) for rgb in plt.colormaps["tab10"].colors]
VALID_TECHS = ["wind", "osw", "pv", "geo"]


class BaseModelStrict(BaseModel):
    """Raise a ValidationError for extra parameters"""

    model_config = {"extra": "forbid"}


class SupplyCurveScenario(BaseModelStrict):
    """Inputs for an individual supply curve scenario"""

    source: Path
    name: str
    color: str = None

    @field_validator("source")
    def expand_user(cls, value):  # noqa: N805
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
    """Container for settings related to specific plots"""

    site_lcoe_max: float = 70
    total_lcoe_max: float = 100


class MapVar(BaseModelStrict):
    """Container for settings related to mapping variables"""

    column: str
    breaks: list[float]
    cmap: str
    legend_title: str


class Config(BaseModelStrict):
    """Configuration settings for creating plots"""

    tech: str
    scenarios: list[SupplyCurveScenario] = []
    plots: Plots = Plots()
    map_vars: list[MapVar] = []
    exclude_maps: list[str] = []
    lcoe_site_col: str = "lcoe_site_usd_per_mwh"
    lcoe_all_in_col: str = "lcoe_all_in_usd_per_mwh"
    cf_col: str = None

    @field_validator("scenarios")
    def default_scenario_colors(cls, value):  # noqa: N805
        """Set default colors for scenarios

        If input scenarios do not have a color set, this function sets
        them to values from the tab10 colormap. This is handled at the
        Config level rather than the SupplyCurveScenario level so that
        the colormap can be incremented for each scenario.

        Parameters
        ----------
        value : list
            List of SupplyCurveScenario models.

        Returns
        -------
        list
            List of SupplyCurveScenarios with default colors set,
            if needed.
        """
        for i, scenario in enumerate(value):
            if scenario.color is None:
                scenario.color = DEFAULT_COLORS[i]

        return value

    @field_validator("tech")
    def valid_tech(cls, value):  # noqa: N805
        """Check that the input value for tech is valid

        Parameters
        ----------
        value : str
            Input value for 'tech'

        Returns
        -------
        str
            Returns the input value (as long as it is one of the valid
            options)

        Raises
        ------
        reVReportsValueError
            A reVReportsValueError will be raised if the input value is
            not a valid option.
        """
        if value not in VALID_TECHS:
            msg = (
                f"Input tech '{value}' is invalid. Valid options are: "
                f"{VALID_TECHS}"
            )
            raise reVReportsValueError(msg)

        return value

    @classmethod
    def from_json(cls, json_path):
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        json_path : path-like
            Path to JSON file containing input settings.

        Returns
        -------
        Config
            Configuration settings.
        """
        with Path(json_path).open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        return cls(**json_data)

    @cached_property
    def scenario_palette(self):
        """
        Get a dictionary mapping scenario names to colors.

        Returns
        -------
        dict
            Dictionary mapping scenario names to colors.
        """
        return {scenario.name: scenario.color for scenario in self.scenarios}
