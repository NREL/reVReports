"""Fonts module"""

from matplotlib import font_manager

from reVReports import PACKAGE_DIR

SANS_SERIF = font_manager.FontEntry(
    fname=PACKAGE_DIR / "assets" / "LocalDejaVuSans.ttf",
    name="LocalDejaVuSans",
)

SANS_SERIF_BOLD = font_manager.FontEntry(
    fname=PACKAGE_DIR / "assets" / "LocalDejaVuSans-Bold.ttf",
    weight="bold",
    name="LocalDejaVuSans",
)
