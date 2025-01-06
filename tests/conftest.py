"""Conftest for Libera Rad unit tests"""
from datetime import datetime, date, timezone
import pytest
import sys
import json
from pathlib import Path
import pandas as pd
from libera_utils.io.manifest import Manifest, ManifestType
# Local
from libera_rad.calibration.calibration_models import LiberaGroundCalibration


@pytest.fixture
def ground_data(test_data_path):
    """Returns a pandas DataFrame of ground data from Dave 11/1"""
    dave_ground_data = test_data_path / "ground_calibration_test_data.csv"
    return pd.read_csv(dave_ground_data)


@pytest.fixture
def calibration_data(calibration_data_path):
    """Returns a dictionary of calibration data"""
    with open(calibration_data_path / "l1b_ground_calibration.json") as f:
        ground_calibration = json.load(f)
    return LiberaGroundCalibration(**ground_calibration)


@pytest.fixture
def calibration_data_path():
    """Returns the Path to the calibration_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent.parent / 'libera_rad' / 'data'


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def generate_input_manifest(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""

    filenames = (test_data_path / "libera_rad_l1b_descriptor_20220909t000000_20220910t000000.h5",
                 test_data_path / "libera_rad_l1b_descriptor_20221010t000000_20221011t000000.h5")

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_files(filenames[0], filenames[1])
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=timezone.utc)
    )

    input_manifest_file_path = input_manifest.write(out_path=tmp_path)

    return input_manifest_file_path
