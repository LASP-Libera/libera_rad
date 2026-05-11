"""Helpers for calibration combiner integration tests."""

from datetime import UTC, date, datetime
from pathlib import Path

import xarray as xr
from cloudpathlib import S3Path
from libera_utils import smart_copy_file, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.product_definition import LiberaDataProductDefinition


def cal_desired_time_range() -> tuple[datetime, datetime]:
    """Return a full-day desired time range for input manifests."""
    return (
        datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )


def copy_cal_input_file(source: Path, dest: Path | S3Path) -> Path | S3Path:
    """Copy a fixture NetCDF into the test input location."""
    return smart_copy_file(source, dest)


def write_cal_netcdf(dataset: xr.Dataset, dest: Path | S3Path) -> None:
    """Write a NetCDF dataset to a local or S3 path."""
    with smart_open(dest, mode="wb") as file_handle:
        dataset.to_netcdf(file_handle, engine="h5netcdf")


def load_cal_netcdf(path: Path | S3Path | str) -> xr.Dataset:
    """Load a NetCDF dataset from a local or S3 path."""
    with smart_open(path, mode="rb") as file_handle:
        return xr.open_dataset(file_handle, engine="h5netcdf").load()


def assert_cal_product_conformance(
    dataset: xr.Dataset,
    product_definitions: dict[DataProductIdentifier, Path],
    product_identifier: DataProductIdentifier,
    expected_product_id: str,
) -> None:
    """Validate a combined calibration product against its product definition."""
    for variable in dataset.variables.values():
        variable.encoding.clear()
    assert dataset.attrs["ProductID"] == expected_product_id
    definition = LiberaDataProductDefinition.from_yaml(product_definitions[product_identifier])
    conformed = definition.enforce_dataset_conformance(dataset)
    errors = definition.check_dataset_conformance(conformed, strict=True)
    assert errors == [], "\n".join(errors[:30])
