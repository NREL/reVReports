"""Fixtures for use across all tests"""

import os
from pathlib import Path

import pytest
import pandas as pd
import geopandas as gpd
from click.testing import CliRunner


LOGGING_META_FILES = {"exceptions.py"}


@pytest.fixture
def assert_message_was_logged(caplog):
    """Assert that a particular (partial) message was logged."""
    caplog.clear()

    def assert_message(msg, log_level=None, clear_records=False):
        """Assert that a message was logged."""
        assert caplog.records

        for record in caplog.records:
            if msg in record.message:
                break
        else:
            err = f"{msg!r} not found in log records"
            raise AssertionError(err)

        # record guaranteed to be defined b/c of "assert caplog.records"
        if log_level:
            assert record.levelname == log_level
        assert record.filename not in LOGGING_META_FILES
        assert record.funcName != "__init__"
        assert "reVReports" in record.name

        if clear_records:
            caplog.clear()

    return assert_message


@pytest.fixture(scope="session")
def test_dir():
    """Directory containing test files"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir(test_dir):
    """Directory containing test data"""
    return test_dir / "data"


@pytest.fixture
def set_to_data_dir(test_data_dir):
    """Temporarily change to the test data directory

    Helpful for handling relative paths to test data
    (e.g., in config files).
    """
    origin = Path().absolute()
    try:
        os.chdir(test_data_dir)
        yield
    finally:
        os.chdir(origin)


@pytest.fixture
def cli_runner():
    """Return a click CliRunner for testing commands"""
    return CliRunner()


@pytest.fixture
def states_gdf(test_data_dir):
    """Return a GeoDataFrame that has states boundaries

    The state boundaries are from ``states.geojson``. To be used as the
    "boundary" layer for maps.map_geodataframe_column() tests.
    """
    state_boundaries_path = (
        test_data_dir / "maps" / "inputs" / "states.geojson"
    )
    states_df = gpd.read_file(state_boundaries_path)
    return states_df.explode(index_parts=True)


@pytest.fixture
def counties_gdf(test_data_dir):
    """Return a GeoDataFrame that has counties boundaries

    The county boundaries are from ``counties.geojson``. To be used as
    used as the "background" layer in maps.map_geodataframe_column()
    tests.
    """

    county_boundaries_path = (
        test_data_dir / "maps" / "inputs" / "counties.geojson"
    )
    counties_df = gpd.read_file(county_boundaries_path)
    counties_df.columns = [s.lower() for s in counties_df.columns]
    counties_df["cnty_fips"] = counties_df["cnty_fips"].astype(int)
    return counties_df


@pytest.fixture
def supply_curve_gdf(test_data_dir):
    """Return a GeoDataFrame that has points from a test supply curve

    Supply curve points consist of results for just a few states.
    """

    supply_curve_path = (
        test_data_dir / "maps" / "inputs" / "map-supply-curve-solar.csv"
    )
    sc_df = pd.read_csv(supply_curve_path)
    return gpd.GeoDataFrame(
        sc_df,
        geometry=gpd.points_from_xy(x=sc_df["longitude"], y=sc_df["latitude"]),
        crs="EPSG:4326",
    )


@pytest.fixture
def background_gdf(test_data_dir):
    """Return a GeoDataFrame that has dissolved state boundaries

    Dissolved state boundaries are from ``states.geojson``. To be used
    as the "background" layer for maps.map_geodataframe_column() tests.
    """
    state_boundaries_path = (
        test_data_dir / "maps" / "inputs" / "states.geojson"
    )
    states_df = gpd.read_file(state_boundaries_path)
    states_dissolved = states_df.union_all()
    return gpd.GeoDataFrame(
        {"geometry": [states_dissolved]}, crs=states_df.crs
    ).explode(index_parts=False)


@pytest.fixture
def county_background_gdf(test_data_dir):
    """Return a GeoDataFrame that has dissolved county boundaries

    Dissolved county boundaries are from ``counties.geojson``. To be
    used as the "background" layer for
    maps.map_geodataframe_column() tests.
    """
    county_boundaries_path = (
        test_data_dir / "maps" / "inputs" / "counties.geojson"
    )
    counties_df = gpd.read_file(county_boundaries_path)
    counties_dissolved = counties_df.union_all()
    return gpd.GeoDataFrame(
        {"geometry": [counties_dissolved]}, crs=counties_df.crs
    ).explode(index_parts=False)
