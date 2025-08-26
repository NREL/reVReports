# Usage Instructions
`revReports` is intended to be used as a command-line tool that can easily and quickly produce standard report-quality graphics for reV supply curves. It currently has the following limitations:
- technologies supported: `wind` and `pv`
- Geographic limitations: `maps` command only works for supply curves for the continental US.
- Number of supply curves: `maps` command can be run for 4 or fewer supply curves. The `plots` command does not have an upper limit, but has only been tested on up to 4 supply curves.

The remainder of this document provides detail on how to use the `reVReports` command-line interface.

## reVReports
The `reVReports` command-line interface can be access following the usage described below.
```cmd
reVReports --help
Usage: reVReports [OPTIONS] COMMAND [ARGS]...

  reVReports command line interface.

Options:
  --version      Show the version and exit.
  -v, --verbose  Flag to turn on debug logging. Default is not verbose.
  --help         Show this message and exit.

Commands:
  maps   Create standard set of report maps for input supply curves.
  plots  Create standard set of report plots for input supply curves.
```

### Maps
The `maps` command creates a set of about 5-10 maps summarizing key results from a set of supply curves. It is currently limited to mapping supply curves for the continental US and 4 or less scenarios.

This command can be run following the usage described below. See the [Configuration Files](#configuration-files) section for additional detail about the `--config-file` argument.
```cmd
reVReports maps --help
Usage: reVReports maps [OPTIONS]

  Create standard set of report maps for input supply curves.

Options:
  -c, --config-file PATH  Path to input configuration JSON file.  [required]
  -o, --out-path PATH     Path to output folder where plots will be saved.
                          Folder will be created if it does not exist.
                          [required]
  -d, --dpi INTEGER       Resolution of output images in dots per inch.
                          Default is 300.
  --help                  Show this message and exit.
```

### Plots
The `plots` command creates a series of about 15 charts, including boxplots, histograms, barcharts, and other summarizing key results from a set of supply curves. There is no limit on the number of supply curves that can be run, but the graphics have only been manually reviewed for 4 or less scenarios.

This command can be run following the usage described below. See the [Configuration Files](#configuration-files) section for additional detail about the `--config-file` argument.
```cmd
reVReports plots --help
Usage: reVReports plots [OPTIONS]

  Create standard set of report plots for input supply curves.

Options:
  -c, --config-file PATH  Path to input configuration JSON file.  [required]
  -o, --out-path PATH     Path to output folder where plots will be saved.
                          Folder will be created if it does not exist.
                          [required]
  -d, --dpi INTEGER       Resolution of output images in dots per inch.
                          Default is 300.
  --help                  Show this message and exit.
```

### Configuration Files
Both the `maps` and `plots` command require an input configuration file that describes, at a minimum, the supply curves to be mapped/plotted. This configuration also has additional options that apply to the `plots` command only, for controlling the colors used for each scenario and for limiting the range of certain graphics.

An example Configuration file is shown below, followed by a description of the inputs.

Example Configuration File
```json
{
  "plots": {
    "site_lcoe_max": 90,
    "total_lcoe_max": 120
  },
  "cf_col": "capacity_factor_ac",
  "lcoe_all_in_col": "lcoe_all_in_usd_per_mwh",
  "lcoe_site_col": "lcoe_site_usd_per_mwh",
  "scenarios": [
    {
      "color": "#8856a7",
      "name": "Open Access",
      "source": "supply_curves/wind/open_access_sample.csv"
    },
    {
      "color": "#8c96c6",
      "name": "Reference Access",
      "source": "supply_curves/wind/reference_access_sample.csv"
    },
    {
      "color": "#b3cde3",
      "name": "Limited Access",
      "source": "supply_curves/wind/limited_access_sample.csv"
    }
  ],
  "map_vars": [
    {
      "column": "cost_spur_usd_per_mw_ac",
      "breaks": [10000, 25000, 50000, 100000, 200000, 500000],
      "cmap": "cool",
      "legend_title": "Spur Line Costs ($/MW)"
    }
  ],
  "tech": "wind"
}
```

Description of Configuration Options:
- `plots`: Optional input that controls some of the settings applied to the `plots` command. Has no effect on `maps`. Optional settings include:
   - `site_lcoe_max`: Maximum limit of site LCOE values to be displayed. Affects multiple plots by either controlling the axis limits or filtering out sites above this value. If not provided, default is `70`.
   - `total_lcoe_max`: Maximum limit of site LCOE values to be displayed. Same effect as previous. If not provided default is `100`.
- `cf_col`: Optional input that specifies which column to use as the capacity factor column in maps and plots. If not specified, the defaul values are `capacity_factor_ac` for `wind` and `osw` and `capacity_factor_dc` for `pv`.
- `lcoe_all_in_col`: Optional input that specified which column to use as the all-in LCOE. If not specified, the column `lcoe_all_in_usd_per_mwh` will be used. If the same value is specified in `lcoe_site_col`, only that LCOE value will be plotted in supply curve plots.
- `lcoe_site_col`: Optional input that specified which column to use as the site-level LCOE. If not specified, the column `lcoe_site_usd_per_mwh` will be used. If the same value is specified in `lcoe_all_in_col`, only that LCOE value will be plotted in supply curve plots.
- `scenarios`: List of supply curve scenarios to be visualized.
    - Each scenario must have the `source` (filepath to supply curve CSV) and `name` (name to be displayed for the scenario).
    - The `color` key is optional - it controls the color to be used for each scenario in the `plots` command, but has no effect on `maps`.
- `map_vars`: List of modified or additional variables to be mapped by the `map` command. If a column specified in this list is part of the default set mapped by the `map` command, the input parameters will be used to overwrite the default breaks, color map, and legend title. If a columns is not part of the default map set, it will be added as an additional output map. The `map_vars` parameter is optional and can be left out of the config or specified as an empty list (`[]`), in which case only the default maps/map settings will be applied.
    - `column`: Name of the column to map.
    - `breaks`: Legend breaks for the map.
    - `cmap`: Colormap to apply. See `matplotlib` colormaps for valid options: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    - `legend_title`: Title to be used for the map legend.
- `exclude_maps`: List of columns to exclude from mapping. Useful for speeding up mapping process or in cases of backwards or cross-compatibility as new columns are added to supply curves. For example, for non-bespoke wind supply curves, specify `"exclude_maps": ["losses_wakes_pct"]` to skip mapping the wake loss column, which may not be present and would result in the `maps` command failing with an error.
- `tech`: Specifies the technology of the input supply curves. Must be either `wind` or `pv`.

### Unpacking Characterizations
The `unpack-characterizations` command can be used to unpack one or more "characterization" columns from a supply curve CSV. These columns typically contain a summary of land-use or other spatial information that characterizes the developable land within each supply curve project site.

The data in these columns are encoded as JSON strings, and thus, not easily accessible for further analysis. Unpacking these JSON strings into useable data is both complicated and slow. The `unpack-characterizations` tool was developed to make this process easier.

An example of a characterization column would be `fed_land_owner`, and a single value in this column could have a value like:
```json
{"2.0": 10.88888931274414, "6.0": 20.11111068725586, "255.0": 2604.860107421875}
```
This JSON string tells us the count of grid cells corresponding to different federal land owners (USFS, BLM, and Non-Federal, respectively) within the developable land for that supply curve project site. Using `unpack-characterizations`, we can unpack this data to give us each of these values in a new, dedicated column, converted to square kilometers:
- `BLM_area_sq_km`: `0.162899997`
- `FS_area_sq_km`: `0.088200003`
- `Non-Federal_area_sq_km`: `21.09936687`

Usage of this command is as follows:
```commandline
Usage: unpack-characterizations [OPTIONS]

  Unpacks characterization data from the input supply curve dataframe,
  converting values from embedded JSON strings to new standalone columns, and
  saves out a new version of the supply curve with these columns included.

Options:
  -i, --supply_curve_csv FILE  Path to bespoke wind supply curve CSV file
                               created by reV  [required]
  -m, --char_map FILE          Path to JSON file storing characterization map
                               [required]
  -o, --out_csv FILE           Path to CSV to store results  [required]
  -c, --cell_size FLOAT        (Optional) Cell size in meters of
                               characterization layers. Default is 90.
  --help                       Show this message and exit.
```

The trickiest part of using `unpack-characterizations` is defining the correct "characterization map" to specify in `-m/--char_map`. This should be a JSON file that defines how to unpack and recast values from the characterization JSON strings to new columns.

Each top-level key in this JSON should be the name of a column of `-i/--supply_curve_csv` containing characterization JSON data. Only the columns you want to unpack need to be included.

The corresponding value should be a dictionary with the following keys: `method`, `recast`, and `lkup` OR `rename`. Details for each are provided below:
- `method`: Must be one of `category`, `sum`, `mean`, or `null`. Note: These correspond to the `method` used for the corresponding layer in the `data_layers` input to reV supply-curve aggregation configuration.
- `recast`: Must be one of `area` or None. This defines how values in the JSON will be recast to new columns. If `area` is specified, they will be converted to area values. If null, they will not be changed and will be passed through as-is.
- `lkup`: This is a dictionary for remapping categories to new column names. Using the `fed_land_owner` example above, it would be: `{"2.0": "FS", "6.0": "BLM", "255.0": "Non-Federal"}`. This follows the same format one could use for ``pandas.rename(columns=lkup)``. This parameter should be used when `method` = `category`. It can also be specified as `null` to skip unpacking of the column.
- `rename`: This is a string indicating what name to use for the new column. This should be used when `method` != `category`.

A valid example of a characterization map can be found [here](tests/data/characterization-map.json) in the test data.
