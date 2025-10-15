# reVReports
The purpose of this library is to make it very simple to create standard, report-quality graphics that summarize the key results from multiple scenarios of reV supply curves. Refer to the [Usage](USAGE.md) for documentation of how to use the library.

## Installation

###  From PyPi

1. Recommended: Setup virtual environment with `conda`/`mamba`:
    ```commandline
    mamba env create -n reVReports
    mamba activate reVReports
    ```

2. Install `reVReports` from PyPi:
    ```commandline
    pip install revreports
    ```
###  From Source

1. Clone the repository
    ```commandline
    git clone git@github.com:NREL/reVReports.git
    ```

2. Move into the local repository
    ```command line
    cd reVReports
    ```

3. Recommended: Setup virtual environment with `conda`/`mamba`:
    ```commandline
    mamba env create -f environment.yml
    mamba activate reVReports
    ```
    Note: You may choose an alternative virtual environment solution; however, installation of dependencies is not guaranteed to work.

4. Install `reVReports`:
    - For users: `pip install .`
    - For developers: `pip install -e '.[dev]'`

5. **Developers Only** Install pre-commit
```commandline
pre-commit install
```

## Developer Notes
To speed up testing, tests can be run in parallel using: `pytest -n auto`.

## Additional Information
NREL Software Record number SWR-25-29.
