# -*- coding: utf-8 -*-
"""fonts module"""
from matplotlib import font_manager

from reVReports import PACKAGE_DIR

SANS_SERIF = font_manager.FontEntry(
    fname=PACKAGE_DIR.joinpath("assets", "LocalDejaVuSans.ttf"), name="LocalDejaVuSans"
)

SANS_SERIF_BOLD = font_manager.FontEntry(
    fname=PACKAGE_DIR.joinpath("assets", "LocalDejaVuSans-Bold.ttf"),
    weight="bold",
    name="LocalDejaVuSans",
)
