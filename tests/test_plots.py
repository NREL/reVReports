# -*- coding: utf-8 -*-
"""Tests for plots module"""
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

from reVReports.plots import (
    configure_matplotlib,
    is_numeric,
    wrap_labels,
    autoscale_x,
    autoscale_y,
    compare_images_approx,
    format_graph,
)


def test_configure_matplotlib():
    """
    Check that configure_matplotlib() correctly changes the formats of negative/minus
    sign.
    """
    # reset any existing settings
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1, -5)
    orig_x_labels = [label.get_text() for label in ax.get_xticklabels()]

    configure_matplotlib()

    new_x_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert orig_x_labels != new_x_labels
    assert new_x_labels == ["-5", "-4", "-3", "-2", "-1"]

    plt.close(fig)


@pytest.mark.parametrize(
    "value,result",
    [
        ("1", True),
        ("-1", True),
        ("0.493", True),
        (1.04, True),
        (1e8, True),
        (-50, True),
        ("1.2.4", False),
        ("1/2", False),
        ("a", False),
        ("one", False),
        ("1-2", False),
    ],
)
def test_is_numeric(value, result):
    """
    Unit test for is_numeric() - loops over value/result pairs defined in fixture.
    """
    assert is_numeric(value) == result


def test_wrap_labels_happy():
    """
    Happy path test for wrap_labels() - check that it correctly breaks up long labels
    based on the specified width.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([None, "Long Label Number 1", "Long Label Number 2", None])
    wrap_labels(ax, width=10)
    new_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert new_labels[1] == "Long Label\nNumber 1"
    assert new_labels[2] == "Long Label\nNumber 2"
    plt.close(fig)


def test_wrap_labels_break_long_words():
    """
    Test wrap_labels() handles the break_long_words argument correctly.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([None, "LongLabelNumber1", "LongLabelNumber2", None])

    wrap_labels(ax, width=10)
    new_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert new_labels[1] == "LongLabelNumber1"
    assert new_labels[2] == "LongLabelNumber2"

    wrap_labels(ax, width=10, break_long_words=True)
    new_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert new_labels[1] == "LongLabelN\number1"
    assert new_labels[2] == "LongLabelN\number2"
    plt.close(fig)


def test_autoscale_x():
    """
    Tests that autoscale_x() correctly scales the x axis to fit the data shown on the
    plot after the y-axis limits have been adjusted.
    """

    limits = (0, 5)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(range(0, 10), range(0, 10))
    ax.set_ylim(*limits)
    assert ax.get_xlim() != limits
    autoscale_x(ax, margin=0)
    assert ax.get_xlim() == limits
    plt.close(fig)


def test_autoscale_y():
    """
    Tests that autoscale_y() correctly scales the y axis to fit the data shown on the
    plot after the x-axis limits have been adjusted.
    """
    limits = (0, 5)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(range(0, 10), range(0, 10))
    ax.set_xlim(*limits)
    assert ax.get_ylim() != limits
    autoscale_y(ax, margin=0)
    assert ax.get_ylim() == limits
    plt.close(fig)


def test_autoscale_y_oob():
    """
    Tests that autoscale_y() gracefully handles case when the there is no data within
    the current plot limits.
    """
    limits = (30, 40)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(range(0, 10), range(0, 10))
    ax.set_xlim(*limits)
    orig_limits = ax.get_ylim()
    assert orig_limits != limits
    autoscale_y(ax, margin=0)
    assert ax.get_ylim() != limits
    assert ax.get_ylim() == orig_limits
    plt.close(fig)


def test_autoscale_x_oob():
    """
    Tests that autoscale_x() gracefully handles case when the there is no data within
    the current plot limits.
    """
    limits = (30, 40)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(range(0, 10), range(0, 10))
    ax.set_ylim(*limits)
    orig_limits = ax.get_xlim()
    assert orig_limits != limits
    autoscale_x(ax, margin=0)
    assert ax.get_xlim() != limits
    assert ax.get_xlim() == orig_limits
    plt.close(fig)


def test_compare_images_approx_match(data_dir):
    """
    Unit test for compare_image_approx() - test that it correctly identifies
    that an image matches itself to a high degree of sensitivity.
    """

    matches, _ = compare_images_approx(
        data_dir.joinpath("compare", "map_1.jpg"),
        data_dir.joinpath("compare", "map_1.jpg"),
        hash_size=64,
        max_diff_pct=0.01,
    )
    assert matches


def test_compare_images_approx_no_match(data_dir):
    """
    Unit test for compare_image_approx() - test that it correctly identifies
    that two very similar image matches don't match when given a high degree of
    sensitivity.
    """

    matches, _ = compare_images_approx(
        data_dir.joinpath("compare", "map_1.jpg"),
        data_dir.joinpath("compare", "map_2.jpg"),
        hash_size=64,
        max_diff_pct=0.01,
    )
    assert not matches


def test_format_graph():
    """
    Catch-all test for format_graph() that applies various formatting options and
    checks that they were applied via introspection of the Axes properties.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ["red", "green", "blue"]
    for i in range(0, 3):
        color = colors[i]
        x = y = np.arange(i * 3, i * 3 + 3) * 1000
        ax.scatter(x, y, c=color, label=color)
    ax.legend()
    format_graph(
        ax,
        xmin=2000,
        xmax=6000,
        xlabel="X Variable",
        ylabel="Y Variable",
        autoscale_to_other_axis=True,
        x_formatter=ticker.StrMethodFormatter("{x:,.1f}"),
        y_formatter=ticker.StrMethodFormatter("{x:,.1f}"),
        legend_frame_on=False,
        move_legend_outside=True,
        drop_legend=False,
        title="Plot Title",
        legend_title="Categories",
    )

    # check that the various settings were applied
    assert ax.get_xlim() == ax.get_ylim() == (2000, 6000)
    assert ax.get_xlabel() == "X Variable"
    assert ax.get_ylabel() == "Y Variable"
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert xticklabels == ["2,000.0", "3,000.0", "4,000.0", "5,000.0", "6,000.0"]
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]
    assert yticklabels == [
        "2,000.0",
        "2,500.0",
        "3,000.0",
        "3,500.0",
        "4,000.0",
        "4,500.0",
        "5,000.0",
        "5,500.0",
        "6,000.0",
    ]
    assert ax.get_title() == "Plot Title"
    legend = ax.get_legend()
    assert legend.get_title().get_text() == "Categories"
    assert legend.get_frame_on() is False

    # check legend position - should be outside to the right of the plot
    # positioned near the center vertically
    legend_box = legend.get_tightbbox()
    plot_box = ax.patch.get_tightbbox()
    assert legend_box.xmin > plot_box.xmax
    assert legend_box.ymin > plot_box.ymin
    assert legend_box.ymax < plot_box.ymax

    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
