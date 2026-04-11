"""Tests for configuration loading."""

import pytest
from pathlib import Path
from src.utils.io import load_config, save_config


def test_load_config_default():
    """Test loading default configuration."""
    config = load_config("configs/default.yaml")
    assert isinstance(config, dict)
    assert "device" in config


def test_load_config_not_found():
    """Test loading non-existent config."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_save_and_load_config(tmp_path):
    """Test saving and loading configuration."""
    config = {"test_key": "test_value", "nested": {"value": 123}}
    config_path = tmp_path / "test_config.yaml"

    save_config(config, str(config_path))
    loaded = load_config(str(config_path))

    assert loaded == config
