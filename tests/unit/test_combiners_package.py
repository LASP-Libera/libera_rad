"""Smoke tests for calibration combiner package imports."""

from libera_rad.calibration.combiners import cal_combine_algorithm, merge_l1a_decoded_datasets


def test_cal_combine_exported_from_combiners_package():
    """Unified cal-combine algorithm should be exported from the package namespace."""
    assert cal_combine_algorithm.__module__.startswith("libera_rad.calibration.combiners")
    assert merge_l1a_decoded_datasets.__module__.startswith("libera_rad.calibration.combiners")
