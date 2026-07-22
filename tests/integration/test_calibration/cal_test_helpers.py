"""Helpers for calibration combiner integration tests."""

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import S3Path
from libera_utils import Manifest, ManifestType, smart_copy_file, smart_open
from libera_utils.io.filenaming import LiberaDataProductFilename, format_from_semantic_version
from libera_utils.io.netcdf import NetcdfEngine

from libera_rad.calibration.constants import CalEventSpec
from libera_rad.config import get_cal_product_definition
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


def build_cal_event_manifest(
    sample_dir: Path,
    filenames: list[str],
    input_dir: Path | S3Path,
    *,
    configuration: dict | None = None,
    spice_kernel_dir: Path | None = None,
) -> Path | S3Path:
    """Copy selected sample files and write a per-event calibration input manifest.

    Parameters
    ----------
    sample_dir : Path
        Directory containing the real L1A sample fixtures.
    filenames : list of str
        Exact Libera filenames to include (one event only).
    input_dir : Path or S3Path
        Destination directory for copied inputs and the manifest.
    configuration : dict, optional
        Manifest configuration (e.g. ``{"use_geo": False}``).
    spice_kernel_dir : Path, optional
        Directory containing AZROT-CK / ELSCAN-CK ``.bc`` files to include.

    Returns
    -------
    Path or S3Path
        Path to the written input manifest.
    """
    copied: list[Path | S3Path] = []
    for name in filenames:
        source = sample_dir / name
        dest = copy_cal_input_file(source, input_dir / name)
        copied.append(dest)

    if spice_kernel_dir is not None:
        for kernel_path in sorted(spice_kernel_dir.glob("LIBERA_SPICE_*.bc")):
            dest = copy_cal_input_file(kernel_path, input_dir / kernel_path.name)
            copied.append(dest)

    manifest = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration=configuration if configuration is not None else {},
    )
    manifest.add_files(*copied)
    start_datetime, end_datetime = cal_desired_time_range()
    manifest.add_desired_time_range(start_datetime=start_datetime, end_datetime=end_datetime)
    return manifest.write(out_path=input_dir)


def assert_azimuth_elevation_positions(
    dataset: xr.Dataset,
    *,
    expect_fill: bool = False,
) -> None:
    """Assert Azimuth_Position / Elevation_Position exist on RAD_SAMPLE_FPE_TIME."""
    for name in ("Azimuth_Position", "Elevation_Position"):
        assert name in dataset, f"Missing {name}"
        assert dataset[name].dims == ("RAD_SAMPLE_FPE_TIME",)
        values = np.asarray(dataset[name].values, dtype=np.float64)
        if expect_fill:
            # Product write may encode _FillValue=-999 as NaN on read-back.
            is_fill = np.isnan(values) | (values == -999.0)
            assert np.all(is_fill), f"{name} expected fill -999 (or NaN after decode)"
        else:
            finite = np.isfinite(values) & (values != -999.0)
            assert np.any(finite), f"{name} expected finite SPICE-derived values"


def assert_companions_within_nom_hk_window(dataset: xr.Dataset) -> None:
    """Assert companion packet/FPE times lie within the NOM-HK event window."""
    if "NOM_HK_PACKET_ICIE_TIME" not in dataset:
        raise AssertionError("Merged calibration product is missing NOM_HK_PACKET_ICIE_TIME")
    t0 = np.datetime64(dataset["NOM_HK_PACKET_ICIE_TIME"].values.min())
    t1 = np.datetime64(dataset["NOM_HK_PACKET_ICIE_TIME"].values.max())

    for name, values in dataset.variables.items():
        if name == "NOM_HK_PACKET_ICIE_TIME":
            continue
        if not (name.endswith("_PACKET_ICIE_TIME") or name.endswith("_FPE_TIME") or name.endswith("FPE_TIME")):
            continue
        if values.size == 0:
            raise AssertionError(f"{name} is empty after event-window slicing")
        vmin = np.datetime64(values.values.min())
        vmax = np.datetime64(values.values.max())
        assert vmin >= t0, f"{name} starts before NOM-HK window: {vmin} < {t0}"
        assert vmax <= t1, f"{name} ends after NOM-HK window: {vmax} > {t1}"


def load_cal_netcdf(path: Path | S3Path | str) -> xr.Dataset:
    """Load a NetCDF dataset using the libera_utils-configured xarray engine."""
    engine = NetcdfEngine.get_from_config()
    with smart_open(path, mode="rb") as file_handle:
        return xr.open_dataset(file_handle, engine=engine).load()


def assert_cal_product_conformance(
    dataset: xr.Dataset,
    product_path: Path | S3Path | str,
    event_spec: CalEventSpec,
) -> None:
    """Validate a combined calibration product against its product definition."""
    for variable in dataset.variables.values():
        variable.encoding.clear()
    assert dataset.attrs["ProductID"] == event_spec.cal_product.value
    assert dataset.attrs["algorithm_version"] == libera_rad_version()
    filename = LiberaDataProductFilename.from_file_path(product_path)
    assert filename.filename_parts.version == format_from_semantic_version(libera_rad_version())
    definition = get_cal_product_definition(event_spec)
    conformed = definition.enforce_dataset_conformance(dataset)
    errors = definition.check_dataset_conformance(conformed, strict=True)
    assert errors == [], "\n".join(errors[:30])
