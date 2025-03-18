# -*- coding: utf-8 -*-
"""
pytest fixtures
"""
from pathlib import Path
import os

import pytest
from click.testing import CliRunner
import pandas as pd
import geopandas as gpd

from reVReports import PACKAGE_DIR

TEST_DATA_DIR = PACKAGE_DIR.parent.joinpath("tests", "data")


@pytest.fixture
def data_dir():
    """Return path to test data directory"""
    return TEST_DATA_DIR


@pytest.fixture
def set_to_data_dir():
    """
    Context manager that temporarily changes the current working directory to the
    test data directory. Helpful for handling relative paths to test data
    (e.g., in config files).
    """
    origin = Path().absolute()
    try:
        os.chdir(TEST_DATA_DIR)
        yield
    finally:
        os.chdir(origin)


@pytest.fixture
def cli_runner():
    """Return a click CliRunner for testing commands"""
    return CliRunner()


@pytest.fixture
def states_gdf():
    """
    Return a geopandas geodataframe that is the states boundaries from
    states.geojson. To be used as the "boundary" layer for
    maps.map_geodataframe_column() tests.
    """

    state_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "maps", "inputs", "states.geojson"
    )
    states_df = gpd.read_file(state_boundaries_path)
    states_singlepart_gdf = states_df.explode(index_parts=True)

    return states_singlepart_gdf


@pytest.fixture
def counties_gdf():
    """
    Return a geopandas geodataframe that is the counties boundaries from
    counties.geojson. To be used as the in
    maps.map_geodataframe_column() tests.
    """

    county_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "maps", "inputs", "counties.geojson"
    )
    counties_df = gpd.read_file(county_boundaries_path)
    counties_df.columns = [s.lower() for s in counties_df.columns]
    counties_df["cnty_fips"] = counties_df["cnty_fips"].astype(int)

    return counties_df


@pytest.fixture
def supply_curve_gdf():
    """
    Return a geopandas geodataframe of points from a test supply curve
    consisting of results for just a few states.
    """

    supply_curve_path = Path(TEST_DATA_DIR).joinpath(
        "maps", "inputs", "map-supply-curve-solar.csv"
    )
    sc_df = pd.read_csv(supply_curve_path)
    sc_gdf = gpd.GeoDataFrame(
        sc_df,
        geometry=gpd.points_from_xy(x=sc_df["longitude"], y=sc_df["latitude"]),
        crs="EPSG:4326",
    )

    return sc_gdf


@pytest.fixture
def background_gdf():
    """
    Return a geopandas geodataframe that is the dissolved boundaries from
    states.geojson. To be used as the "background" layer for
    maps.map_geodataframe_column() tests.
    """

    state_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "maps", "inputs", "states.geojson"
    )
    states_df = gpd.read_file(state_boundaries_path)
    states_dissolved = states_df.union_all()
    states_dissolved_gdf = gpd.GeoDataFrame(
        {"geometry": [states_dissolved]}, crs=states_df.crs
    ).explode(index_parts=False)

    return states_dissolved_gdf


@pytest.fixture
def county_background_gdf():
    """
    Return a geopandas geodataframe that is the dissolved boundaries from
    counties.geojson. To be used as the "background" layer for
    maps.map_geodataframe_column() tests.
    """

    county_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "maps", "inputs", "counties.geojson"
    )
    counties_df = gpd.read_file(county_boundaries_path)
    counties_dissolved = counties_df.union_all()
    counties_dissolved_gdf = gpd.GeoDataFrame(
        {"geometry": [counties_dissolved]}, crs=counties_df.crs
    ).explode(index_parts=False)

    return counties_dissolved_gdf
