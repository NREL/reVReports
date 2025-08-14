# reVReports
The purpose of this library is to make it very simple to create standard, report-quality graphics that summarize the key results from multiple scenarios of reV supply curves. Refer to the [Usage](USAGE.md) for documentation of how to use the library.

## Installation
1. Clone the repo
    ```commandline
    git clone git@github.nrel.gov:GDS/reVReports.git
    ```

2. Move into the local repo
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
