"""Conftest for Libera Rad unit tests"""

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from libera_utils.io.manifest import Manifest, ManifestType

# Local
from libera_rad.calibration.calibration_models import LiberaGroundCalibration


@pytest.fixture(scope="session")
def ground_data(test_data_path):
    """Returns a pandas DataFrame of ground data from Dave 11/1"""
    dave_ground_data = test_data_path / "ground_calibration_test_data.csv"
    return pd.read_csv(dave_ground_data)


@pytest.fixture(scope="session")
def calibration_data(calibration_data_path):
    """Returns a dictionary of calibration data"""
    with open(calibration_data_path / "l1b_ground_calibration.json") as f:
        ground_calibration = json.load(f)
    return LiberaGroundCalibration(**ground_calibration)


@pytest.fixture(scope="session")
def calibration_data_path():
    """Returns the Path to the calibration_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent.parent / "libera_rad" / "data"


@pytest.fixture(scope="session")
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_l1a_cal_data_path(test_data_path):
    """Returns the Path to the test_l1a_cal_data directory"""
    return test_data_path / "cal_l1a_data"


@pytest.fixture(scope="session")
def test_dynamic_kernels_path(test_data_path):
    """Returns the Path to the test geolocation kernels directory"""
    return test_data_path / "dynamic_kernels"


@pytest.fixture(scope="session")
def test_integration_data_path(test_data_path):
    """Returns the Path to the integration test l1b directory"""
    return test_data_path / "l1b_integration_data"


@pytest.fixture
def generate_input_manifest(tmp_path, test_integration_data_path):
    """Generating test manifest from the data in test_data"""
    # Radiometer L1A data
    l1a_radiometer_test_file = (
        test_integration_data_path
        / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"
    )

    # Housekeeping L1A Data
    l1a_housekeeping_test_file = (
        test_integration_data_path / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"
    )

    # SPICE Kernels - elevation
    spice_kernel_elevation_file = (
        test_integration_data_path / "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26016220328.bc"
    )
    # SPICE Kernels - azimuth
    spice_kernel_azimuth_file = (
        test_integration_data_path / "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26016220138.bc"
    )
    # SPICE Kernels - jpss spk
    spice_kernel_jpss_spk_file = (
        test_integration_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp"
    )
    # SPICE Kernels - jpss ck
    spice_kernel_jpss_ck_file = (
        test_integration_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc"
    )

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_files(
        l1a_radiometer_test_file,
        l1a_housekeeping_test_file,
        spice_kernel_elevation_file,
        spice_kernel_azimuth_file,
        spice_kernel_jpss_spk_file,
        spice_kernel_jpss_ck_file,
    )
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )

    input_manifest_file_path = input_manifest.write(out_path=tmp_path)

    return str(input_manifest_file_path)
