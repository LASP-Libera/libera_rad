"""End-to-end cal-combine tests against real TRIMMED L1A sample fixtures."""

from pathlib import Path

import pytest
from libera_utils import Manifest

from libera_rad.calibration.combiners.cal_combine import algorithm
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID, LIBERA_CAL_OBSID_ENV
from tests.integration.test_calibration.cal_test_helpers import (
    assert_cal_product_conformance,
    assert_companions_within_nom_hk_window,
    build_cal_event_manifest,
    load_cal_netcdf,
)

_SAMPLE_ONE = Path("sample_one")
_SAMPLE_TWO = Path("sample_two")

_REAL_EVENTS = [
    pytest.param(
        _SAMPLE_TWO,
        512,
        [
            "LIBERA_L1A_NOM-HK-GAIN-TRIMMED_V5-8-5RC1_20280212T005109_20280212T005929_R26199220207.nc",
            "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5RC1_20280212T005115_20280212T005929_R26163174745.nc",
            "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5RC1_20280212T005518_20280212T005919_R26163174745.nc",
        ],
        {},
        id="gain-512",
    ),
    pytest.param(
        _SAMPLE_TWO,
        320,
        [
            "LIBERA_L1A_NOM-HK-LWC-TEMP1-TRIMMED_V5-8-5RC1_20280212T000127_20280212T000735_R26199220207.nc",
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280212T000050_20280212T020052_R26163174745.nc",
            "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5RC1_20280212T000059_20280212T020011_R26163174745.nc",
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280212T000147_20280212T020019_R26163174745.nc",
        ],
        {},
        id="lwc-320",
    ),
    pytest.param(
        _SAMPLE_ONE,
        257,
        [
            "LIBERA_L1A_NOM-HK-SWC-405NM-TRIMMED_V5-8-5RC1_20280213T030640_20280213T031036_R26199213122.nc",
            "LIBERA_L1A_CAL-SAMPLE-DECODED_V5-8-5RC1_20280213T030614_20280213T031034_R26163174745.nc",
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280213T020114_20280213T040014_R26163174745.nc",
            "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5RC1_20280213T020149_20280213T040001_R26163174745.nc",
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280213T020154_20280213T035926_R26163174745.nc",
        ],
        {},
        id="swc-257",
    ),
    pytest.param(
        _SAMPLE_ONE,
        385,
        [
            "LIBERA_L1A_NOM-HK-SOLAR-TOT-PRI-TRIMMED_V5-8-5RC1_20280213T021710_20280213T021830_R26199213122.nc",
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280213T020114_20280213T040014_R26163174745.nc",
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280213T020154_20280213T035926_R26163174745.nc",
        ],
        {"solar_cal_face": 1, "event_pass_index": 1, "source_obsids": [385]},
        id="solar-385",
    ),
    pytest.param(
        _SAMPLE_ONE,
        386,
        [
            "LIBERA_L1A_NOM-HK-SOLAR-LW-PRI-TRIMMED_V5-8-5RC1_20280213T035840_20280213T040000_R26199213122.nc",
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5RC1_20280213T020114_20280213T040014_R26163174745.nc",
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5RC1_20280213T020154_20280213T035926_R26163174745.nc",
        ],
        {"solar_cal_face": 1, "event_pass_index": 2, "source_obsids": [386]},
        id="solar-386",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("path_type", ["Local", "S3"], indirect=True)
@pytest.mark.parametrize(("sample_subdir", "obsid", "filenames", "solar_attrs"), _REAL_EVENTS)
def test_cal_combine_real_sample_event(
    test_l1a_cal_data_path,
    cal_io_paths,
    monkeypatch,
    sample_subdir: Path,
    obsid: int,
    filenames: list[str],
    solar_attrs: dict,
):
    """Run cal-combine on one real-event manifest and validate cropped output."""
    input_dir, output_dir = cal_io_paths
    event_spec = CAL_EVENT_BY_OBSID[obsid]
    sample_dir = test_l1a_cal_data_path / sample_subdir

    manifest_path = build_cal_event_manifest(sample_dir, filenames, input_dir)
    monkeypatch.setenv("PROCESSING_PATH", str(output_dir))
    monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, str(obsid))

    output_manifest_path = algorithm(manifest_path)
    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1

    output_file = output_manifest.files[0].filename
    dataset = load_cal_netcdf(output_file)
    assert_cal_product_conformance(dataset, output_file, event_spec)
    assert_companions_within_nom_hk_window(dataset)

    for attr_name, expected in solar_attrs.items():
        assert dataset.attrs[attr_name] == expected
