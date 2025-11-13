"""reVReports"""  # noqa

from pathlib import Path

from reVReports.version import __version__

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR.joinpath("datasets")
