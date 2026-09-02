"""Tests for bundled libera_rad product definitions."""

import tomllib
from pathlib import Path

import yaml

from libera_rad.config import (
    CAL_FAMILY_PRODUCT_DEFINITIONS,
    data_path,
    product_config_path,
)
from libera_rad.version import version as libera_rad_version


def _product_definition_paths() -> list[Path]:
    return sorted({product_config_path, *CAL_FAMILY_PRODUCT_DEFINITIONS.values()})


def test_product_definition_algorithm_version_is_dynamic():
    """algorithm_version must be null in YAML so it is injected from the package at write time."""
    for definition_path in _product_definition_paths():
        definition = yaml.safe_load(definition_path.read_text())
        assert definition["attributes"]["algorithm_version"] is None, definition_path.name


def test_package_version_matches_pyproject():
    """The installed package version must match pyproject.toml."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_version = tomllib.load(pyproject_file)["project"]["version"]
    assert libera_rad_version() == pyproject_version


def test_all_bundled_product_definitions_exist():
    """Every configured product definition path must exist under libera_rad/data."""
    for definition_path in _product_definition_paths():
        assert definition_path.exists(), definition_path
        assert definition_path.parent == data_path
