"""Tests for the l1b algorithm"""
# Standard
from argparse import Namespace
from datetime import datetime, timezone, timedelta
# Installed
import pytest
import xarray as xr
from cloudpathlib import AnyPath
from libera_utils.io.manifest import Manifest
# Local
from libera_rad.pass_through import algorithm as pass_through_algorithm
from libera_rad.pass_through import (write_dummy_spice_jpss_files,
                                     write_dummy_spice_az_el_files,
                                     write_dummy_l1b_files)


def test_write_spice_jpss(tmp_path):
    """Testing the writing of the spice jpss file"""
    data = xr.DataArray([1, 2, 3, 4, 5])
    data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    data.attrs['Incoming_manifest_name'] = "test_manifest"
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=1)
    spk_file_path, ck_file_path = write_dummy_spice_jpss_files(data, str(tmp_path), start_time, end_time)
    assert AnyPath(spk_file_path).exists()
    assert AnyPath(ck_file_path).exists()


def test_write_spice_az_el(tmp_path):
    """Testing the writing of the spice jpss file"""
    data = xr.DataArray([1, 2, 3, 4, 5])
    data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    data.attrs['Incoming_manifest_name'] = "test_manifest"
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=1)
    az_file_path, el_file_path = write_dummy_spice_az_el_files(data, str(tmp_path), start_time, end_time)
    assert AnyPath(az_file_path).exists()
    assert AnyPath(el_file_path).exists()


@pytest.mark.parametrize(
    ("instrument_type"),
    [
        "CAM",
        "RAD"
    ])
def test_write_l1b(tmp_path, instrument_type):
    """Testing the writing of the l1b file"""
    data = xr.DataArray([1, 2, 3, 4, 5])
    data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    data.attrs['Incoming_manifest_name'] = "test_manifest"
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=1)
    l1b_file_path = write_dummy_l1b_files(data, str(tmp_path), instrument_type, start_time, end_time)
    assert AnyPath(l1b_file_path).exists()


@pytest.mark.parametrize(
    ("processing_step"),
    [
        "spice-jpss",
        "spice-azel",
        "l1b-cam",
        "l1b-rad"
    ])
def test_pass_through_algorithm(generate_input_manifest, monkeypatch, tmp_path, processing_step):
    """Testing the algorithm to generate output manifests"""

    monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
    algorithm_inputs = Namespace(manifest=str(generate_input_manifest), processing_step=processing_step)
    output_manifest_path = pass_through_algorithm(algorithm_inputs)

    output_manifest_obj = Manifest.from_file(output_manifest_path)
    for file in output_manifest_obj.files:
        assert AnyPath(file['filename']).exists()
