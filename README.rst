**********
reVReports
**********

|License| |Zenodo| |PythonV| |PyPi| |Ruff| |Pixi| |SWR|

.. |PythonV| image:: https://badge.fury.io/py/reVReports.svg
    :target: https://pypi.org/project/reVReports/

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/reVReports.svg
    :target: https://pypi.org/project/reVReports/

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

.. |Pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh

.. |SWR| image:: https://img.shields.io/badge/SWR--25--29_-blue?label=NREL
    :alt: Static Badge

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17633670.svg
    :target: https://doi.org/10.5281/zenodo.17633670

.. inclusion-intro


reVReports is a tool that aims  to make it very simple to create standard,
report-quality graphics that summarize the key results from multiple scenarios
of reV supply curves.


Installing reVReports
=====================
The quickest way to install reVReports for users is from PyPi:

.. code-block:: bash

    pip install reVReports

If you would like to install and run reVReports from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NREL/reVReports.git; cd reVReports
    pixi run reVReports

For detailed instructions, see the `installation documentation <https://nrel.github.io/reVReports/misc/installation.html>`_.


Quickstart
==========
To run a quick reVReports demo, use:

.. code-block:: shell

    pixi run geo-demo

This will generate sample map outputs using example reV geothermal supply curve outputs.

For more information on running ``reVReports``, see
`Usage <https://github.com/NREL/reVReports/blob/main/USAGE.md>`_.


Development
===========
Please see the `Development Guidelines <https://nrel.github.io/reVReports/dev/index.html>`_
if you wish to contribute code to this repository.
