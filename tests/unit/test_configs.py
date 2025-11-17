"""Tests for Config class"""

import pytest
from pydantic import ValidationError

from reVReports.configs import Config, DEFAULT_COLORS


def test_config_happy(test_data_dir):
    """Happy path test for Config class

    This test makes sure it can load settings from a json
    file and resulting properties can be accessed.
    """
    json_path = test_data_dir / "config_wind_bespoke.json"
    config = Config.from_json(json_path)
    assert config.tech == "wind"
    assert len(config.scenarios) == 3
    assert config.plots.site_lcoe_max == 90.0
    assert config.plots.total_lcoe_max == 120.0
    assert config.lcoe_site_col == "lcoe_site_usd_per_mwh"
    assert config.lcoe_all_in_col == "lcoe_all_in_usd_per_mwh"
    assert config.cf_col == "capacity_factor_ac"
    assert config.map_vars == []


def test_config_defaults(test_data_dir):
    """Test config class will set defaults for optional inputs"""
    json_path = test_data_dir / "config_wind_bespoke_missing_props.json"
    config = Config.from_json(json_path)
    assert config.plots.site_lcoe_max == 70
    assert config.plots.total_lcoe_max == 100
    assert config.lcoe_site_col == "lcoe_site_usd_per_mwh"
    assert config.lcoe_all_in_col == "lcoe_all_in_usd_per_mwh"
    assert config.cf_col is None
    assert config.map_vars == []
    for i, scenario in enumerate(config.scenarios):
        assert scenario.color == DEFAULT_COLORS[i]


def test_config_strict(test_data_dir):
    """Test  Config class will raise ValidationError for extra params"""
    json_path = test_data_dir / "config_misplaced_params.json"
    with pytest.raises(
        ValidationError, match=r".*4 validation errors for Config.*"
    ):
        Config.from_json(json_path)


def test_config_map_vars(test_data_dir):
    """Test Config class will properly load map_vars properties"""
    json_path = test_data_dir / "config_pv_map_vars.json"
    config = Config.from_json(json_path)
    assert len(config.map_vars) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
