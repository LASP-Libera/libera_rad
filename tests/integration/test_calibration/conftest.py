"""Shared fixtures for calibration combiner integration tests."""

import pytest
from cloudpathlib import S3Path


@pytest.fixture
def path_type(request):
    """Passthrough for indirect parameterization of local vs S3 storage."""
    return request.param


@pytest.fixture
def cal_io_paths(path_type, tmp_path, create_mock_bucket):
    """Return input and output directories for local or mocked-S3 calibration tests."""
    if path_type == "S3":
        input_bucket = create_mock_bucket()
        output_bucket = create_mock_bucket()
        input_dir = S3Path(f"s3://{input_bucket.name}/input/")
        output_dir = S3Path(f"s3://{output_bucket.name}/output/")
        input_dir.mkdir(parents=True, exist_ok=True)
    else:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
    return input_dir, output_dir
