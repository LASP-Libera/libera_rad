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
        Manifest configuration dictionary.
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


#: Minimum share of Az/El samples SPICE must resolve.
#:
#: Not 1.0: companions are trimmed a whole packet at a time, so the edge packets contribute
#: samples just outside the motor CK coverage and those take the -999 fill. Observed across the
#: four Az/El sample events, 42-72 samples per product fall outside, worst case 0.9955. A floor
#: rather than "any finite value" so a run where SPICE resolved almost nothing still fails.
_MIN_AZEL_FINITE_FRACTION = 0.99


def assert_azimuth_elevation_positions(dataset: xr.Dataset) -> None:
    """Assert Azimuth_Position / Elevation_Position are on FPE time and mostly SPICE-derived."""
    for name in ("Azimuth_Position", "Elevation_Position"):
        assert name in dataset, f"Missing {name}"
        assert dataset[name].dims == ("RAD_SAMPLE_FPE_TIME",)
        values = np.asarray(dataset[name].values, dtype=np.float64)
        finite = np.isfinite(values) & (values != -999.0)
        assert finite.mean() >= _MIN_AZEL_FINITE_FRACTION, (
            f"{name} is only {finite.mean():.4f} SPICE-derived ({int((~finite).sum())} of {finite.size} samples filled)"
        )


#: Slack allowed when bounding companion sample times by the NOM-HK window.
#:
#: Companions are trimmed a whole packet at a time, so the edge packets of the retained run
#: contribute samples on both sides of the window boundary. A RAD-SAMPLE packet carries 50
#: samples at ~150 Hz, so one packet's span is well under a second; this bound catches a
#: companion that is genuinely off the event while tolerating that edge.
_SAMPLE_WINDOW_SLACK = np.timedelta64(2, "s")


def assert_companions_within_nom_hk_window(dataset: xr.Dataset) -> None:
    """Assert companion *sample* times lie within the NOM-HK event window, plus one packet.

    Sample time is what selects packets, so it is the axis the window actually bounds.
    ``*_PACKET_ICIE_TIME`` is deliberately not bounded: an FPE-skewed packet whose samples land
    inside the window is kept, and its own timestamp can sit well outside it (a RAD-SAMPLE
    packet stamped 16 s past the window has been observed in ground-test data).

    Sample-to-packet integrity is covered by :func:`assert_sample_packet_axes_agree`.
    """
    if "NOM_HK_PACKET_ICIE_TIME" not in dataset:
        raise AssertionError("Merged calibration product is missing NOM_HK_PACKET_ICIE_TIME")
    t0 = np.datetime64(dataset["NOM_HK_PACKET_ICIE_TIME"].values.min())
    t1 = np.datetime64(dataset["NOM_HK_PACKET_ICIE_TIME"].values.max())

    checked_any = False
    for name, values in dataset.variables.items():
        if not str(name).endswith("_FPE_TIME"):
            continue
        if values.size == 0:
            raise AssertionError(f"{name} is empty after event-window slicing")
        checked_any = True
        vmin = np.datetime64(values.values.min())
        vmax = np.datetime64(values.values.max())
        assert vmin >= t0 - _SAMPLE_WINDOW_SLACK, f"{name} starts before NOM-HK window: {vmin} < {t0}"
        assert vmax <= t1 + _SAMPLE_WINDOW_SLACK, f"{name} ends after NOM-HK window: {vmax} > {t1}"

    assert checked_any, "Merged calibration product has no sample time variables to bound"


def assert_sample_packet_axes_agree(dataset: xr.Dataset) -> None:
    """Assert each sample axis agrees with its packet axis, via ``*_packet_index``.

    Every sample group carries a ``<group>_packet_index`` recording which packet each sample
    came from. Trimming renumbers it from 0 against the surviving packet axis, so in a written
    calibration product it must be exactly ``repeat(arange(n_packets), samples_per_packet)``.

    A packet and sample axis sliced to different packet runs shows up here and nowhere else:
    the indices stay in range for the pre-trim axis and point at the wrong packets.
    """
    sample_dims = [str(dim) for dim in dataset.dims if str(dim).endswith("_FPE_TIME")]
    assert sample_dims, "Merged calibration product has no sample dimensions to check"

    for sample_dim in sample_dims:
        group = sample_dim[: -len("_FPE_TIME")]
        packet_dim = f"{group}_PACKET"
        assert packet_dim in dataset.dims, f"{sample_dim} has no matching {packet_dim} dimension"
        n_packets = dataset.sizes[packet_dim]
        n_samples = dataset.sizes[sample_dim]
        assert n_samples % n_packets == 0, (
            f"{sample_dim} has {n_samples} samples, not an exact multiple of {packet_dim}={n_packets}; "
            f"the packet and sample axes describe different sets of packets"
        )

        index_var = f"{group}_packet_index"
        assert index_var in dataset.variables, (
            f"{sample_dim} has no {index_var}; the sample-to-packet validation link is missing"
        )
        packet_index = np.asarray(dataset[index_var].values)
        assert packet_index.dtype == np.int64, f"{index_var} should be int64, got {packet_index.dtype}"
        np.testing.assert_array_equal(
            packet_index,
            np.repeat(np.arange(n_packets, dtype=np.int64), n_samples // n_packets),
            err_msg=f"{index_var} is not a 0-based index into {packet_dim} (size {n_packets})",
        )


def assert_filename_covers_data(dataset: xr.Dataset, product_path: Path | S3Path | str) -> None:
    """Assert the product filename's time range spans every *sample* time variable in the file.

    Every family names its file from ``NOM_HK_PACKET_ICIE_TIME``: NOM-HK defines the
    calibration event and is the only time base shared across families. ``*_PACKET_ICIE_TIME``
    is deliberately excluded from this check — whole-packet retention and FPE skew let a
    companion's packet timestamps sit outside the stamped range even though its samples do not.
    """
    parts = LiberaDataProductFilename.from_file_path(product_path).filename_parts
    start = np.datetime64(parts.utc_start.replace(tzinfo=None), "ns")
    end = np.datetime64(parts.utc_end.replace(tzinfo=None), "ns")

    for name, values in dataset.variables.items():
        if not str(name).endswith("_FPE_TIME"):
            continue
        vmin = np.datetime64(values.values.min(), "ns")
        vmax = np.datetime64(values.values.max(), "ns")
        # Filenames are truncated to whole seconds, and a retained edge packet contributes
        # samples just outside the window, so allow a small amount of slack per edge.
        assert vmin >= start - _SAMPLE_WINDOW_SLACK, f"filename starts at {start}, after {name} begins at {vmin}"
        assert vmax <= end + _SAMPLE_WINDOW_SLACK, f"filename ends at {end}, before {name} ends at {vmax}"


def assert_input_files_provenance(dataset: xr.Dataset, expected_inputs: list[str]) -> None:
    """Assert ``input_files`` names the granules the product was actually built from.

    It must list the L1A companions and SPICE kernels this run consumed. The merge inherits
    attributes from the NOM-HK input, whose own ``input_files`` holds the L0 CCSDS packet files
    from its decode, so ``ccsds_*`` entries mean the attribute was never rewritten.
    """
    input_files = [str(name) for name in np.atleast_1d(dataset.attrs["input_files"])]

    l0_leftovers = [name for name in input_files if name.startswith("ccsds_")]
    assert not l0_leftovers, f"input_files still carries L0 packet files from the NOM-HK decode: {l0_leftovers}"

    missing = [name for name in expected_inputs if name not in input_files]
    assert not missing, f"input_files is missing {missing}; got {input_files}"


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
