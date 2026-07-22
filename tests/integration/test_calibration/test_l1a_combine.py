"""Merge/prefix checks for ``merge_l1a_decoded_datasets`` using real sample fixtures."""

from pathlib import Path

import xarray as xr

from libera_rad.calibration.combiners.l1a_combine import merge_l1a_decoded_datasets

_SAMPLE_ONE = Path("sample_one")
_SAMPLE_TWO = Path("sample_two")
_PACKET_SLICE = 10

# Real sample filenames keyed for readable test setup.
_SWC_NOM_HK = "LIBERA_L1A_NOM-HK-SWC-405NM-TRIMMED_V5-8-5RC1_20280213T030640_20280213T031036_R26199213122.nc"
_SWC_CAL_SAMPLE = "LIBERA_L1A_CAL-SAMPLE-DECODED_V5-8-5RC1_20280213T030614_20280213T031034_R26163174745.nc"
_SWC_RAD_SAMPLE = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280213T020114_20280213T040014_R26163174745.nc"
_SWC_PEC = "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5RC1_20280213T020149_20280213T040001_R26163174745.nc"
_SWC_PEV = "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280213T020154_20280213T035926_R26163174745.nc"

_LWC_NOM_HK = "LIBERA_L1A_NOM-HK-LWC-TEMP1-TRIMMED_V5-8-5RC1_20280212T000127_20280212T000735_R26199220207.nc"
_LWC_RAD_SAMPLE = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280212T000050_20280212T020052_R26163174745.nc"
_LWC_PEC = "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5RC1_20280212T000059_20280212T020011_R26163174745.nc"
_LWC_PEV = "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280212T000147_20280212T020019_R26163174745.nc"

_GAIN_NOM_HK = "LIBERA_L1A_NOM-HK-GAIN-TRIMMED_V5-8-5RC1_20280212T005109_20280212T005929_R26199220207.nc"
_GAIN_RAD_FULL = "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5RC1_20280212T005115_20280212T005929_R26163174745.nc"
_GAIN_CAL_FULL = "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5RC1_20280212T005518_20280212T005919_R26163174745.nc"

_SOLAR_NOM_HK = "LIBERA_L1A_NOM-HK-SOLAR-TOT-PRI-TRIMMED_V5-8-5RC1_20280213T021710_20280213T021830_R26199213122.nc"
_SOLAR_RAD_SAMPLE = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280213T020114_20280213T040014_R26163174745.nc"
_SOLAR_PEV = "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280213T020154_20280213T035926_R26163174745.nc"


def _open_packet_slice(path: Path, n_packets: int = _PACKET_SLICE) -> xr.Dataset:
    """Load a real L1A sample and keep only the first ``n_packets`` packets."""
    dataset = xr.open_dataset(path).load()
    return dataset.isel(PACKET=slice(0, min(n_packets, dataset.sizes["PACKET"])))


def _assert_prefixed_packet_dims(merged: xr.Dataset, expected: dict[str, int]) -> None:
    assert "PACKET" not in merged.dims
    for dim_name, size in expected.items():
        assert dim_name in merged.sizes, f"missing dimension {dim_name}"
        assert merged.sizes[dim_name] == size


def _expected_packet_sizes(prefixes: tuple[str, ...], datasets: list[xr.Dataset]) -> dict[str, int]:
    return {f"{prefix}_PACKET": ds.sizes["PACKET"] for prefix, ds in zip(prefixes, datasets, strict=True)}


def test_sw_cal_l1a_combine(test_l1a_cal_data_path):
    sample = test_l1a_cal_data_path / _SAMPLE_ONE
    datasets = [
        _open_packet_slice(sample / _SWC_CAL_SAMPLE),
        _open_packet_slice(sample / _SWC_RAD_SAMPLE),
        _open_packet_slice(sample / _SWC_NOM_HK),
        _open_packet_slice(sample / _SWC_PEC),
        _open_packet_slice(sample / _SWC_PEV),
    ]
    prefixes = ("CAL_SAMPLE", "RAD_SAMPLE", "NOM_HK", "PEC_SW_STAT", "PEV_SW_STAT")
    merged = merge_l1a_decoded_datasets(datasets)
    _assert_prefixed_packet_dims(merged, _expected_packet_sizes(prefixes, datasets))


def test_lw_cal_l1a_combine(test_l1a_cal_data_path):
    sample = test_l1a_cal_data_path / _SAMPLE_TWO
    datasets = [
        _open_packet_slice(sample / _LWC_RAD_SAMPLE),
        _open_packet_slice(sample / _LWC_NOM_HK),
        _open_packet_slice(sample / _LWC_PEC),
        _open_packet_slice(sample / _LWC_PEV),
    ]
    prefixes = ("RAD_SAMPLE", "NOM_HK", "PEC_SW_STAT", "PEV_SW_STAT")
    merged = merge_l1a_decoded_datasets(datasets)
    _assert_prefixed_packet_dims(merged, _expected_packet_sizes(prefixes, datasets))


def test_gain_cal_l1a_combine(test_l1a_cal_data_path):
    sample = test_l1a_cal_data_path / _SAMPLE_TWO
    datasets = [
        _open_packet_slice(sample / _GAIN_RAD_FULL),
        _open_packet_slice(sample / _GAIN_CAL_FULL),
        _open_packet_slice(sample / _GAIN_NOM_HK),
    ]
    prefixes = ("RAD_FULL", "CAL_FULL", "NOM_HK")
    merged = merge_l1a_decoded_datasets(datasets)
    _assert_prefixed_packet_dims(merged, _expected_packet_sizes(prefixes, datasets))


def test_solar_cal_combine(test_l1a_cal_data_path):
    sample = test_l1a_cal_data_path / _SAMPLE_ONE
    datasets = [
        _open_packet_slice(sample / _SOLAR_RAD_SAMPLE),
        _open_packet_slice(sample / _SOLAR_NOM_HK),
        _open_packet_slice(sample / _SOLAR_PEV),
    ]
    prefixes = ("RAD_SAMPLE", "NOM_HK", "PEV_SW_STAT")
    merged = merge_l1a_decoded_datasets(datasets)
    _assert_prefixed_packet_dims(merged, _expected_packet_sizes(prefixes, datasets))
