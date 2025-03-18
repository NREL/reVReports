# -*- coding: utf-8 -*-
"""Tests for CLI"""
# pylint: disable=use-implicit-booleaness-not-comparison
import pytest
from pydantic import ValidationError

from reVReports.configs import Config, DEFAULT_COLORS


def test_config_happy(data_dir):
    """
    Happy path test for Config class - make sure it can load settings from a json
    file and resulting properties can be accessed.
    """
    json_path = data_dir.joinpath("config_wind_bespoke.json")
    config = Config.from_json(json_path)
    assert config.tech == "wind"
    assert len(config.scenarios) == 3
    assert config.plots.site_lcoe_max == 90.0
    assert config.plots.total_lcoe_max == 120.0
    assert config.lcoe_site_col == "lcoe_site_usd_per_mwh"
    assert config.lcoe_all_in_col == "lcoe_all_in_usd_per_mwh"
    assert config.cf_col == "capacity_factor_ac"
    assert config.map_vars == []


def test_config_defaults(data_dir):
    """
    Test that the Config class will set defaults for optional input properties.
    """
    json_path = data_dir.joinpath("config_wind_bespoke_missing_props.json")
    config = Config.from_json(json_path)
    assert config.plots.site_lcoe_max == 70
    assert config.plots.total_lcoe_max == 100
    assert config.lcoe_site_col == "lcoe_site_usd_per_mwh"
    assert config.lcoe_all_in_col == "lcoe_all_in_usd_per_mwh"
    assert config.cf_col is None
    assert config.map_vars == []
    for i, scenario in enumerate(config.scenarios):
        assert scenario.color == DEFAULT_COLORS[i]


def test_config_strict(data_dir):
    """
    Test that the Config class will raise a ValidationError if the config has extra
    parameters.
    """
    json_path = data_dir.joinpath("config_misplaced_params.json")
    with pytest.raises(ValidationError, match=".*4 validation errors for Config.*"):
        Config.from_json(json_path)


def test_config_map_vars(data_dir):
    """
    Test that the Config class will properly load map_vars properties.
    """
    json_path = data_dir.joinpath("config_pv_map_vars.json")
    config = Config.from_json(json_path)
    assert len(config.map_vars) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
