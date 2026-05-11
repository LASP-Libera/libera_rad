"""Smoke tests for calibration combiner package imports."""

import importlib

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


def test_combiners_modules_are_importable():
    """All combiner modules should import successfully from their canonical paths."""
    module_names = [
        "libera_rad.calibration.combiners.gain_combiner",
        "libera_rad.calibration.combiners.l1a_cal_event_utils",
        "libera_rad.calibration.combiners.l1a_combine",
        "libera_rad.calibration.combiners.lw_cal_combiner",
        "libera_rad.calibration.combiners.solar_cal_combiner",
        "libera_rad.calibration.combiners.sw_combiner",
    ]
    for module_name in module_names:
        importlib.import_module(module_name)
