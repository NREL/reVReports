"""Plotting utilities"""

import textwrap

import matplotlib
from matplotlib import style as mplstyle
from matplotlib import ticker
import seaborn as sns
import numpy as np
import PIL
import imagehash

from reVReports.fonts import SANS_SERIF
from reVReports.exceptions import reVReportsValueError

DPI = 300
DEFAULT_RC_PARAMS = {
    "patch.edgecolor": (1, 1, 1, 0.5),
    "patch.linewidth": 0.1,
}
DEFAULT_FORMATTER = ticker.StrMethodFormatter("{x:,.0f}")
NO_OUTLINE_RC_PARAMS = DEFAULT_RC_PARAMS.copy()
NO_OUTLINE_RC_PARAMS.update(
    {"patch.edgecolor": "none", "patch.force_edgecolor": False}
)

SMALL_SIZE = 12
SMALL_MEDIUM_SIZE = 13
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
RC_FONT_PARAMS = {
    "font.size": SMALL_SIZE,
    "axes.titlesize": BIGGER_SIZE,
    "axes.labelsize": BIGGER_SIZE,
    "xtick.labelsize": MEDIUM_SIZE,
    "ytick.labelsize": MEDIUM_SIZE,
    "legend.fontsize": MEDIUM_SIZE,
    "figure.titlesize": BIGGER_SIZE,
    "figure.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelpad": 8,
    "axes.labelweight": "bold",
    "font.family": SANS_SERIF.name,
}


def configure_matplotlib():
    """Adjust settings of matplotlib for faster plotting"""
    # set to use ascii hyphen rather than unicode minus
    matplotlib.rcParams["axes.unicode_minus"] = False
    mplstyle.use("fast")


def is_numeric(s):
    """Check whether a variable is numeric

    Parameters
    ----------
    s : Any
        Variable to check.

    Returns
    -------
    bool
        True if the number is numeric, false if not.
    """
    try:
        _ = float(s)
    except (ValueError, TypeError):
        return False

    return True


def wrap_labels(ax, width, break_long_words=False):
    """Wrap tick labels of the x axis

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        (sub-)Plot axes of matplotlib image.
    width : int
        Maximum width of each line of the wrapped label (in number of
        characters).
    break_long_words : bool, optional
        If True, long words may be hyphenated and split across lines.
        By default ``False``, which will not split long words.
    """

    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels, rotation=0)


def autoscale_y(ax, margin=0.1):
    """Rescale the y-axis to fit the subset of data that is visible

    Rescaling is done based on the current limited of the x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        (sub-)Plot axes of matplotlib image.
    margin : float, optional
        Padding to use in setting the y limits. By default 0.1, which
        means 10% of the range of the data subset's y-values.
    """

    def get_bottom_top(x_data, y_data):
        """Derive the y-limits to be used based on the visible data"""
        x_lim_low, x_lim_high = ax.get_xlim()
        y_displayed = y_data[((x_data >= x_lim_low) & (x_data <= x_lim_high))]
        if len(y_displayed) == 0:
            return np.inf, -np.inf
        height = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * height
        top = np.max(y_displayed) + margin * height
        return bot, top

    if len(ax.patches) > 0:
        msg = "Support for plots with patches has not been implemented."
        raise NotImplementedError(msg)

    bot, top = np.inf, -np.inf
    lines = ax.get_lines()
    collections = ax.collections
    if len(lines) > 0:
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            new_bot, new_top = get_bottom_top(x_data, y_data)
            bot = min(bot, new_bot)
            top = max(top, new_top)
    elif len(collections) > 0:
        for collection in collections:
            x_data = collection.get_offsets()[:, 0].data
            y_data = collection.get_offsets()[:, 1].data
            new_bot, new_top = get_bottom_top(x_data, y_data)
            bot = min(bot, new_bot)
            top = max(top, new_top)
    else:
        msg = "No lines or collections found in plot."
        raise reVReportsValueError(msg)

    if bot != np.inf:
        ax.set_ylim(ymin=bot)
    if top != -np.inf:
        ax.set_ylim(ymax=top)


def autoscale_x(ax, margin=0.1):
    """Rescale the x-axis to fit the subset of data that is visible

    Rescaling is based on the current limited of the y-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        (sub-)Plot axes of matplotlib image.
    margin : float, optional
        Padding to use in setting the x limits. By default 0.1, which
        means 10% of the range of the data subset's x-values.
    """

    def get_left_right(x_data, y_data):
        """Derive the x-limits to be used based on the visible data."""
        y_lim_low, y_lim_high = ax.get_ylim()
        x_displayed = x_data[((y_data >= y_lim_low) & (y_data <= y_lim_high))]
        if len(x_displayed) == 0:
            return np.inf, -np.inf
        width = np.max(x_displayed) - np.min(x_displayed)
        left = np.min(x_displayed) - margin * width
        right = np.max(x_displayed) + margin * width
        return left, right

    if len(ax.patches) > 0:
        msg = "Support for plots with patches has not been implemented."
        raise NotImplementedError(msg)

    left, right = np.inf, -np.inf
    lines = ax.get_lines()
    collections = ax.collections
    if len(lines) > 0:
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            new_left, new_right = get_left_right(x_data, y_data)
            left = min(left, new_left)
            right = max(right, new_right)
    elif len(collections) > 0:
        for collection in collections:
            x_data = collection.get_offsets()[:, 0].data
            y_data = collection.get_offsets()[:, 1].data
            new_left, new_right = get_left_right(x_data, y_data)
            left = min(left, new_left)
            right = max(right, new_right)
    else:
        msg = "No lines or collections found in plot."
        raise reVReportsValueError(msg)

    if left != np.inf:
        ax.set_xlim(xmin=left)
    if right != -np.inf:
        ax.set_xlim(xmax=right)


def format_graph(  # noqa: PLR0913, PLR0917
    graph,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xlabel=None,
    ylabel=None,
    autoscale_to_other_axis=False,
    x_formatter=DEFAULT_FORMATTER,
    y_formatter=DEFAULT_FORMATTER,
    legend_frame_on=False,
    move_legend_outside=False,
    drop_legend=False,
    title=None,
    legend_title=None,
):
    """
    Formatter for single seaborn plots. Does not work for facet plots or
    seaborn.objects.Plots.

    Parameters
    ----------
    graph : matplotlib.axes.Axes
        (sub-)Plot axes of matplotlib image, created by seaborn
        (e.g., seaborn.Boxplot).
    xmin : float, optional
        Minimum x-value to display. By default None, which will have no
        effect on the graph.
    xmax : float, optional
        Maximum x-value to display. By default None, which will have no
        effect on the graph.
    ymin : float, optional
        Minimum y-value to display. By default None, which will have no
        effect on the graph.
    ymax : _type_, optional
        Maximum y-value to display. By default None, which will have no
        effect on the graph.
    xlabel : str, optional
        Label to use for the x-axis, by default None which will result
        in no x-axis label.
    ylabel : str, optional
        Label to use for the y-axis, by default None which will result
        in no y-axis label.
    autoscale_to_other_axis : bool, optional
        If True, and the limits of one of the axes is changed (e.g.,
        using ``xmax`` or ``ymax``, etc.), the other axis will be
        rescaled to adjust to the subset of data displayed in the graph.
    x_formatter : matplotlib.ticker.StrMethodFormatter, optional
        Formatter to use for x-axis ticks. Default is
        ticker.StrMethodFormatter("{x:,.0f}"), which will use commas for
        thousands. Specify ``None`` to apply no formatting.
    y_formatter : matplotlib.ticker.StrMethodFormatter, optional
        Formatter to use for y-axis ticks. Default is
        ticker.StrMethodFormatter("{x:,.0f}"), which will use commas for
        thousands. Specify ``None`` to apply no formatting.
    legend_frame_on : bool, optional
        If True, and there is a legend, it will include a legend frame
        (i.e., outline). By default False, which hides the legend frame.
    move_legend_outside : bool, optional
        If True and there is a legend, move the legend outside the plot
        (to the right side of the plot, centered vertically).
        By default ``False``, which leaves the legend position
        unchanged.
    drop_legend : bool, optional
        If True and there is a legend, the legend will be
        removed/hidden. By default ``False``, which leaves the legend
        visible.
    title : str, optional
        Title to use for the graph. By default None, which displays no
        title.
    legend_title : str, optional
        Title to use for the legend. By default None, which leaves the
        legend title unchanged.

    Returns
    -------
    matplotlib.axes.Axes
        Formatted graph.
    """

    graph.set(xlim=(xmin, xmax))
    graph.set(ylim=(ymin, ymax))

    _autoscale_axes(graph, xmax, ymax, autoscale_to_other_axis)

    graph.set(xlabel=xlabel)
    graph.set(ylabel=ylabel)

    _format_axes(graph, x_formatter, y_formatter)
    _format_legend(
        graph, legend_frame_on, move_legend_outside, drop_legend, legend_title
    )

    if title is not None:
        graph.set_title(title)

    return graph


def compare_images_approx(
    image_1_path, image_2_path, hash_size=12, max_diff_pct=0.25
):
    """Check if two images match approximately

    Parameters
    ----------
    image_1_path : pathlib.Path
        File path to first image.
    image_2_path : pathlib.Path
        File path to first image.
    hash_size : int, default=12
        Size of the image hashes that will be used for image comparison,
        by default 12. Increase to make the check more precise, decrease
        to make it more approximate.
    max_diff_pct : float, default=0.05
        Tolerance for the amount of difference allowed (0.05 = 5%).
        Increase to allow for a larger delta between the image hashes,
        decrease to make the check stricter and require a smaller delta
        between the image hashes.

    Returns
    -------
    bool
        Returns true if the images match approximately, false if not.
    """

    expected_hash = imagehash.phash(
        PIL.Image.open(image_1_path), hash_size=hash_size
    )
    out_hash = imagehash.phash(
        PIL.Image.open(image_2_path), hash_size=hash_size
    )

    max_diff_bits = int(np.ceil(hash_size**2 * max_diff_pct))

    diff = expected_hash - out_hash
    matches = diff <= max_diff_bits
    pct_diff = float(diff) / hash_size**2

    return matches, pct_diff


def _autoscale_axes(graph, xmax, ymax, autoscale_to_other_axis):
    if ymax is None and xmax is not None and autoscale_to_other_axis is True:
        autoscale_y(graph)
    if xmax is None and ymax is not None and autoscale_to_other_axis is True:
        autoscale_x(graph)


def _format_axes(graph, x_formatter, y_formatter):
    if x_formatter:
        ticks = graph.xaxis.get_ticklabels()
        if len(ticks) > 0 and is_numeric(ticks[0].get_text()):
            graph.axes.xaxis.set_major_formatter(x_formatter)
    if y_formatter:
        ticks = graph.yaxis.get_ticklabels()
        if len(ticks) > 0 and is_numeric(ticks[0].get_text()):
            graph.axes.yaxis.set_major_formatter(y_formatter)


def _format_legend(
    graph, legend_frame_on, move_legend_outside, drop_legend, legend_title
):
    if graph.legend_ is None:
        return

    if drop_legend:
        graph.legend_.remove()
        return

    graph.legend_.set_frame_on(legend_frame_on)

    if move_legend_outside:
        sns.move_legend(graph, "center left", bbox_to_anchor=(1, 0.5))

    graph.legend_.set_title(legend_title)
