"""Maps utilities tests"""

import tempfile
from pathlib import Path

import pytest
import mapclassify as mc
import numpy as np
import matplotlib.pyplot as plt
import geoplot as gplt
from shapely.geometry import box

from reVReports.utilities.plots import compare_images_approx
from reVReports.utilities.maps import YBFixedBounds, map_geodataframe_column


def test_YBFixedBounds_happy():  # noqa: N802
    """
    Happy path test for YBFixedBounds. Check that it correctly resets
    max() and min() methods to return preset values rather than actual min
    and max of the input array.
    """

    data = np.arange(1, 10)

    preset_max = 15
    preset_min = 0
    yb = YBFixedBounds(data, preset_max=15, preset_min=0)

    assert data.max() != preset_max
    assert data.min() != preset_min
    assert yb.max() == preset_max
    assert yb.min() == preset_min


def test_YBFixedBounds_mapclassify():  # noqa: N802
    """
    Test YBFixedBounds works as expected when used to overwrite the yb
    property of a mapclassify classifier.
    """

    data = np.arange(1, 90)
    breaks = [20, 40, 60, 80, 100]
    scheme = mc.UserDefined(data, bins=breaks)
    preset_max = scheme.k
    present_min = 0

    assert scheme.yb.max() < scheme.k
    scheme.yb = YBFixedBounds(
        scheme.yb, preset_max=preset_max, preset_min=present_min
    )
    assert scheme.yb.max() == preset_max
    assert scheme.yb.min() == present_min


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_happy(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Happy path test for map_geodataframe_column. Test that when run
    with basic inputs and default settings, the output image matches
    the expected image.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir) / "happy_map.png"
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_styling(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column() produces expected output image when
    various styling parameters are passed.
    """

    col_name = "area_sq_km"
    color_map = "GnBu"

    breaks = [15, 20, 25, 30, 40]
    map_extent = states_gdf.buffer(0.05).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Styling Map",
            legend_title=col_name.title(),
            background_df=background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent,
            layer_kwargs={"s": 4, "linewidth": 0, "marker": "o"},
            legend_kwargs={
                "marker": "o",
                "frameon": True,
                "bbox_to_anchor": (1, 0),
                "loc": "upper left",
            },
        )
        plt.tight_layout()

        out_png_name = "styling_map.png"
        out_png = Path(tempdir) / out_png_name
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_repeat(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that running map_geodataframe_column twice exactly the same produces
    the same output. This covers a previously discovered bug where the legend
    symbols would change from squares to circles for the second map in a
    sequence.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
        )
        plt.tight_layout()
        plt.close(g.figure)

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir) / out_png_name
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_no_legend(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column function produces a map without a legend
    when the legend argument is specified as False.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            legend=False,
        )
        plt.tight_layout()

        out_png_name = "no_legend.png"
        out_png = Path(tempdir) / "no_legend.png"
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_boundaries_kwargs(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column function produces a map with correctly
    styled boundaries when boundaries_kwargs are passed.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            boundaries_kwargs={
                "linewidth": 2,
                "zorder": 1,
                "edgecolor": "black",
            },
        )
        plt.tight_layout()

        out_png_name = "boundaries_kwargs.png"
        out_png = Path(tempdir) / "boundaries_kwargs.png"
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_polygons(
    test_data_dir,
    supply_curve_gdf,
    county_background_gdf,
    states_gdf,
    counties_gdf,
):
    """
    Test that map_geodataframe_column() produces expected output image
    for a polygon input layer.
    """

    county_area_sq_km_df = (
        supply_curve_gdf.groupby("cnty_fips")["area_sq_km"].sum().reset_index()
    )
    county_capacity_gdf = counties_gdf.merge(
        county_area_sq_km_df, how="inner", on="cnty_fips"
    )

    col_name = "area_sq_km"
    color_map = "YlOrRd"

    breaks = [250, 350, 450, 550]
    map_extent = county_background_gdf.buffer(0.05).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:
        g = map_geodataframe_column(
            county_capacity_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Polygons Map",
            legend_title=col_name.title(),
            background_df=county_background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent,
            layer_kwargs={"edgecolor": "gray", "linewidth": 0.5},
            boundaries_kwargs={
                "linewidth": 1,
                "zorder": 2,
                "edgecolor": "black",
            },
            legend_kwargs={
                "marker": "s",
                "frameon": False,
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left",
            },
        )
        plt.tight_layout()

        out_png_name = "polygons_map.png"
        out_png = Path(tempdir) / out_png_name
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_existing_ax(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column correctly plots on an existing GeoAxes
    when provided as an input to the function.
    """
    col_name = "area_sq_km"

    center_lon, center_lat = box(
        *supply_curve_gdf.total_bounds.tolist()
    ).centroid.coords[0]
    with tempfile.TemporaryDirectory() as tempdir:
        fig, ax = plt.subplots(
            ncols=2,
            nrows=1,
            figsize=(13, 4),
            subplot_kw={
                "projection": gplt.crs.AlbersEqualArea(
                    central_longitude=center_lon, central_latitude=center_lat
                )
            },
        )
        for panel in ax.ravel().tolist():
            panel = map_geodataframe_column(  # noqa: PLW2901
                supply_curve_gdf,
                col_name,
                background_df=background_gdf,
                boundaries_df=states_gdf,
                ax=panel,
            )

        out_png_name = "map_2panels.png"
        out_png = Path(tempdir) / "map_2panels.png"
        fig.savefig(out_png, dpi=75)
        plt.close(fig)

        expected_png = test_data_dir / "maps" / "outputs" / out_png_name

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png, hash_size=64, max_diff_pct=0.1
        )
        msg = (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )
        assert images_match, msg


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
