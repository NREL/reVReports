# -*- coding: utf-8 -*-
"""Data module: functions and classes for modifying/augmenting/translating data"""
ORDERED_REGIONS = [
    "Pacific",
    "California",
    "Southwest",
    "Mountain",
    "Great Plains",
    "Great Lakes",
    "Northeast",
    "South Central",
    "Southeast",
]


NREL_REGIONS = {
    "Oregon": "Pacific",
    "Washington": "Pacific",
    "Colorado": "Mountain",
    "Idaho": "Mountain",
    "Montana": "Mountain",
    "Wyoming": "Mountain",
    "Iowa": "Great Plains",
    "Kansas": "Great Plains",
    "Missouri": "Great Plains",
    "Minnesota": "Great Plains",
    "Nebraska": "Great Plains",
    "North Dakota": "Great Plains",
    "South Dakota": "Great Plains",
    "Illinois": "Great Lakes",
    "Indiana": "Great Lakes",
    "Michigan": "Great Lakes",
    "Ohio": "Great Lakes",
    "Wisconsin": "Great Lakes",
    "Connecticut": "Northeast",
    "New Jersey": "Northeast",
    "New York": "Northeast",
    "Maine": "Northeast",
    "New Hampshire": "Northeast",
    "Massachusetts": "Northeast",
    "Pennsylvania": "Northeast",
    "Rhode Island": "Northeast",
    "Rhode Island and Providence Plantations": "Northeast",
    "Vermont": "Northeast",
    "California": "California",
    "Arizona": "Southwest",
    "Nevada": "Southwest",
    "New Mexico": "Southwest",
    "Utah": "Southwest",
    "Arkansas": "South Central",
    "Louisiana": "South Central",
    "Oklahoma": "South Central",
    "Texas": "South Central",
    "Alabama": "Southeast",
    "Delaware": "Southeast",
    "District of Columbia": "Southeast",
    "Florida": "Southeast",
    "Georgia": "Southeast",
    "Kentucky": "Southeast",
    "Maryland": "Southeast",
    "Mississippi": "Southeast",
    "North Carolina": "Southeast",
    "South Carolina": "Southeast",
    "Tennessee": "Southeast",
    "Virginia": "Southeast",
    "West Virginia": "Southeast",
}


def augment_sc_df(
    df,
    scenario_name,
    scenario_number,
    tech,
    lcoe_all_in_col="lcoe_all_in_usd_per_mwh",
    cf_col=None,
):
    """
    Augment an input supply curve dataframe with additional columns needed for
    standard plots. This function is intended for use on supply curves created with
    reV version ≥ 0.14.5.

    Parameters
    ----------
    df : pandas.DataFrame
        Input supply curve dataframe.
    scenario_name : str
        Name of the scenario associated with the supply curve
    scenario_number : int
        Number of the scenario. This is used to control ordering if multiple scenarios
        are being plotted. Scenarios will be ordered based on ascending order of
        this input value (i.e., if you want this scenario to plot first, set
        scenario_number to 0 and subsequent scenarios to 1, 2, etc.)
    tech : str
        The technology of the input supply curves. Must be either `wind` or `pv`.
    lcoe_all_in_col : str, default="lcoe_all_in_usd_per_mwh"
        Column name that represents the All-in LCOE cost values.
        By default, ``"lcoe_all_in_usd_per_mwh"``.
    cf_col : str, default=None
        Name of column storing the capacity factor values. By default, this is set to
        None, and defaults will be automatically set based on the input technology
        (e.g., "capacity_factor_dc" for "pv", "capacity_factor_ac" for "wind", "osw",
        and "geo")

    Returns
    -------
    pandas.DataFrame
        Augmented dataframe with additional columns for the Scenario, scenario_number,
        and multiple additional quantitative results.
    """

    # add standard named capacity_mw column
    if tech == "wind":
        capacity_col = "capacity_ac_mw"
        default_cf_col = "capacity_factor_ac"
    elif tech == "osw":
        capacity_col = "capacity_ac_mw"
        default_cf_col = "capacity_factor_ac"
    elif tech == "pv":
        capacity_col = "capacity_dc_mw"
        default_cf_col = "capacity_factor_dc"
    elif tech == "geo":
        capacity_col = "capacity_ac_mw"
        default_cf_col = "capacity_factor_ac"
    else:
        raise ValueError(
            f"Invalid input: tech={tech}. "
            "Valid options are: ['wind', 'pv', 'osw', 'geo']."
        )

    if cf_col is None:
        cf_col = default_cf_col

    df["capacity_mw"] = df[capacity_col]
    df["capacity_factor"] = df[cf_col]
    df["Scenario"] = scenario_name
    df["scenario_number"] = scenario_number
    df["capacity_density"] = df["capacity_mw"] / df["area_developable_sq_km"]

    df.sort_values(by=[lcoe_all_in_col], ascending=True, inplace=True)
    df["lcot_pct_lcoe"] = df["lcot_usd_per_mwh"] / df[lcoe_all_in_col] * 100

    df["cost_interconnection_usd_per_mw"] = (
        df["cost_spur_usd_per_mw_ac"] + df["cost_poi_usd_per_mw_ac"]
    )
    df["cumul_capacit_gw"] = df["capacity_mw"].cumsum() / 1000
    df["cumul_aep_twh"] = df["annual_energy_site_mwh"].cumsum() / 1000 / 1000
    df["cumul_area_sq_km"] = df["area_developable_sq_km"].cumsum()
    df["nrel_region"] = df["state"].map(NREL_REGIONS)
    df["nrel_region_cumul_capacit_gw"] = (
        df.groupby("nrel_region")["capacity_mw"].cumsum() / 1000
    )

    return df


def check_files_match(pattern, dir_1_path, dir_2_path):
    """
    Verify that the files in two folders match. Files are filtered by the specified
    pattern and the contents of the two folders are searched recursively. Only the
    names of the files and their relative paths within the fodlers are compared -
    contents of the files are not checked.

    Parameters
    ----------
    pattern : str
        File pattern used for filtering files in the specified folders.
    dir_1_path : pathlib.Path
        Path to the first directory.
    dir_2_path : pathlib.Path
        Path to the second directory.
    Returns
    -------
    tuple
        Returns a tuple with two elements: the first element is a boolean indicating
        whether the files in the two folders match and the second is a list of
        differences between the two folders (if applicable). If the files match, the
        list will be empty.
    """

    output_files = [f.relative_to(dir_1_path) for f in dir_1_path.rglob(pattern)]
    expected_output_files = [
        f.relative_to(dir_2_path) for f in dir_2_path.rglob(pattern)
    ]

    difference = list(
        set(output_files).symmetric_difference(set(expected_output_files))
    )
    if len(difference) == 0:
        return True, []

    return False, difference
