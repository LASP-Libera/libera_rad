"""Conftest for Libera Rad unit tests"""
from datetime import datetime, date, timezone
import pytest
import sys
from pathlib import Path
from libera_utils.io.manifest import Manifest, ManifestType


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def generate_input_manifest(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""

    filenames = (test_data_path / "libera_rad_l1b_descriptor_20220909t000000_20220910t000000.h5",
                 test_data_path / "libera_rad_l1b_descriptor_20221010t000000_20221011t000000.h5")

    input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_file_to_manifest(filenames[0])
    input_manifest.add_file_to_manifest(filenames[1])
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=timezone.utc)
    )

    input_manifest_file_path = input_manifest.write(outpath=tmp_path)

    return input_manifest_file_path
