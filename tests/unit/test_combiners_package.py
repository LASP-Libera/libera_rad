"""Smoke tests for calibration combiner package imports."""

from libera_rad.calibration.combiners import (
    gain_algorithm,
    lw_algorithm,
    merge_l1a_decoded_datasets,
    solar_algorithm,
    sw_algorithm,
)


def test_algorithms_exported_from_combiners_package():
    """Combiner algorithms should be exported from the package namespace."""
    assert gain_algorithm.__module__.startswith("libera_rad.calibration.combiners")
    assert lw_algorithm.__module__.startswith("libera_rad.calibration.combiners")
    assert solar_algorithm.__module__.startswith("libera_rad.calibration.combiners")
    assert sw_algorithm.__module__.startswith("libera_rad.calibration.combiners")
    assert merge_l1a_decoded_datasets.__module__.startswith("libera_rad.calibration.combiners")


def test_combiners_package_imports_without_error():
    """Importing the combiners package should not raise."""
    import libera_rad.calibration.combiners  # noqa: F401
