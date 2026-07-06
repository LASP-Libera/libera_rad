"""Helpers for calibration combiner integration tests."""

from datetime import UTC, date, datetime
from pathlib import Path

import xarray as xr
from cloudpathlib import S3Path
from libera_utils import smart_copy_file, smart_open
from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename, format_from_semantic_version
from libera_utils.io.netcdf import NetcdfEngine, write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path

from libera_rad.version import version as libera_rad_version


def cal_desired_time_range() -> tuple[datetime, datetime]:
    """Return a full-day desired time range for input manifests."""
    return (
        datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )


def copy_cal_input_file(source: Path, dest: Path | S3Path) -> Path | S3Path:
    """Copy a fixture NetCDF into the test input location."""
    return smart_copy_file(source, dest)


def write_nom_hk_fixture(dataset: xr.Dataset, output_dir: Path | S3Path) -> Path | S3Path:
    """Write a modified NOM-HK L1A fixture via write_libera_data_product."""
    result = write_libera_data_product(
        data_product_definition=get_l1a_product_definition_path(LiberaApid.icie_nom_hk),
        data=dataset,
        output_path=output_dir,
        time_variable="PACKET_ICIE_TIME",
        strict=True,
    )
    return result.path


def load_cal_netcdf(path: Path | S3Path | str) -> xr.Dataset:
    """Load a NetCDF dataset using the libera_utils-configured xarray engine."""
    engine = NetcdfEngine.get_from_config()
    with smart_open(path, mode="rb") as file_handle:
        return xr.open_dataset(file_handle, engine=engine).load()


def assert_cal_product_conformance(
    dataset: xr.Dataset,
    product_path: Path | S3Path | str,
    product_definitions: dict[DataProductIdentifier, Path],
    product_identifier: DataProductIdentifier,
    expected_product_id: str,
) -> None:
    """Validate a combined calibration product against its product definition."""
    for variable in dataset.variables.values():
        variable.encoding.clear()
    assert dataset.attrs["ProductID"] == expected_product_id
    assert dataset.attrs["algorithm_version"] == libera_rad_version()
    filename = LiberaDataProductFilename.from_file_path(product_path)
    assert filename.filename_parts.version == format_from_semantic_version(libera_rad_version())
    definition = LiberaDataProductDefinition.from_yaml(product_definitions[product_identifier])
    conformed = definition.enforce_dataset_conformance(dataset)
    errors = definition.check_dataset_conformance(conformed, strict=True)
    assert errors == [], "\n".join(errors[:30])
