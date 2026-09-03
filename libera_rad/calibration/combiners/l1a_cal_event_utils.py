"""L1A calibration event utilities — manifest load, slicing, kernels, and event building.

Shared helpers for ObsID-dispatched calibration products:

- Load decoded L1A datasets from a calibration input manifest
- Select the family TRIMMED NOM-HK granule that defines the event
- Confirm NOM-HK ObsIDs against ``LIBERA_CAL_OBSID``
- Select companion products and slice them to the NOM-HK event window
- Generate this event's motor CKs from AXIS-SAMPLE and query them for Azimuth/Elevation
- Merge the selected streams into one calibration event dataset
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import S3Path
from libera_utils import Manifest, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.kernel_maker import create_kernel_from_l1a
from libera_utils.l1a.packet_slicing import find_sample_dims, slice_l1a_dataset_to_time_window
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad import geolocation
from libera_rad.calibration.combiners.l1a_combine import merge_l1a_decoded_datasets
from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    LIBERA_CAL_OBSID_ENV,
    CalEventSpec,
)
from libera_rad.version import version as libera_rad_version

logger = logging.getLogger(__name__)

#: Motor CK products generated per event from AXIS-SAMPLE (furnish order: AZ then EL).
_GENERATED_SPICE_CAL_AZEL: tuple[DataProductIdentifier, ...] = (
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_el_ck,
)

#: Sample axis of the AXIS-SAMPLE-DECODED product, carrying the 200 Hz encoder angles.
_AXIS_SAMPLE_TIME = "AXIS_SAMPLE_ICIE_TIME"

#: Families that receive SPICE-derived Azimuth_Position / Elevation_Position, keyed by the
#: family TRIMMED ProductID. GAIN is a full-rate merge with no motor attitude.
_AZEL_POSITION_FAMILIES: frozenset[DataProductIdentifier] = frozenset(
    {
        DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed,
        DataProductIdentifier.l1a_icie_nom_hk_lwc_family_trimmed,
        DataProductIdentifier.l1a_icie_nom_hk_solar_family_trimmed,
    }
)

#: Global attribute listing the granules this product was built from.
_INPUT_FILES_ATTR = "input_files"
_RAD_SAMPLE_FPE_TIME = "RAD_SAMPLE_FPE_TIME"
_AZIMUTH_POSITION = "Azimuth_Position"
_ELEVATION_POSITION = "Elevation_Position"


def read_all_cal_input_data(input_manifest: Manifest) -> dict[str, xr.Dataset]:
    """Load the decoded L1A datasets from a calibration input manifest.

    Every manifest entry is an L1A granule. cal-combine has no SPICE inputs — it generates the
    motor CKs it needs from the AXIS-SAMPLE-DECODED L1A input (see :func:`event_azel_kernels`).

    Parameters
    ----------
    input_manifest : Manifest
        Calibration combiner input manifest.

    Returns
    -------
    dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.

    Raises
    ------
    Exception
        If any manifest entry cannot be opened as an L1A NetCDF granule.
    """
    all_data: dict[str, xr.Dataset] = {}

    for i, file_info in enumerate(input_manifest.files):
        logger.info("Reading file %d/%d: %s", i + 1, len(input_manifest.files), file_info.filename)
        try:
            with smart_open(file_info.filename) as file_handle:
                LiberaDataProductFilename.from_file_path(file_info.filename)
                dataset = xr.open_dataset(file_handle, decode_times=True).load()
                all_data[file_info.filename] = dataset
                logger.info("Successfully loaded dataset: %s", file_info.filename)
        except Exception:
            logger.error("Failed to process file %s", file_info.filename, exc_info=True)
            raise

    logger.info("Successfully loaded %d datasets", len(all_data))
    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data


def attach_azimuth_elevation_positions(
    cal_event: xr.Dataset,
    dynamic_kernel_sources: Sequence[str | Path | S3Path],
) -> xr.Dataset:
    """Attach SPICE-derived ``Azimuth_Position`` / ``Elevation_Position`` on FPE time.

    Parameters
    ----------
    cal_event : xr.Dataset
        Merged SWC/LWC/SOLAR calibration event dataset.
    dynamic_kernel_sources : sequence of str, Path, or S3Path
        AZROT-CK and ELSCAN-CK paths to furnish, in that order. In cal-combine these are the
        event-scoped kernels built by :func:`event_azel_kernels`; the parameter stays generic so
        the query is testable against any kernel pair.

    Returns
    -------
    xr.Dataset
        ``cal_event`` with ``Azimuth_Position`` and ``Elevation_Position`` assigned.

    Raises
    ------
    ValueError
        If ``RAD_SAMPLE_FPE_TIME`` is missing or kernel sources are empty.
    """
    if _RAD_SAMPLE_FPE_TIME not in cal_event:
        raise ValueError(f"Calibration event dataset is missing {_RAD_SAMPLE_FPE_TIME}")

    timestamps = np.asarray(cal_event[_RAD_SAMPLE_FPE_TIME].values)
    if not dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required for Azimuth_Position / Elevation_Position")

    with KernelManager() as km:
        km.load_libera_dynamic_kernels(dynamic_kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)
        azimuth, elevation = geolocation.calculate_azimuth_elevation_for_timestamps(km, timestamps)

    cal_event = cal_event.assign(
        {
            _AZIMUTH_POSITION: (_RAD_SAMPLE_FPE_TIME, np.asarray(azimuth, dtype=np.float32)),
            _ELEVATION_POSITION: (_RAD_SAMPLE_FPE_TIME, np.asarray(elevation, dtype=np.float32)),
        }
    )
    return cal_event


@contextmanager
def event_azel_kernels(
    axis_sample: xr.Dataset,
    t0: np.datetime64,
    t1: np.datetime64,
) -> Iterator[list[PathType]]:
    """Build AZROT-CK and ELSCAN-CK covering the NOM-HK event window.

    The AXIS-SAMPLE granule is trimmed to ``[t0, t1]`` before the kernels are built, so each CK
    covers the calibration event and nothing else. Trimming is driven by
    :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window`, which selects whole
    packets on ``AXIS_SAMPLE_ICIE_TIME`` — the same sample-time rule every other companion is
    trimmed by, so the kernels and the merged streams are cut on one consistent basis.

    The kernels are run-local intermediates rebuildable from the AXIS-SAMPLE granule, so they are
    written to a temporary directory and deleted when the context exits. Only that granule is
    recorded as a product input.

    Parameters
    ----------
    axis_sample : xr.Dataset
        Decoded AXIS-SAMPLE-DECODED dataset for the day containing the event.
    t0 : np.datetime64
        Event window start (inclusive), from NOM-HK.
    t1 : np.datetime64
        Event window end (inclusive), from NOM-HK.

    Yields
    ------
    list of PathType
        Generated kernel paths in furnish order (AZROT then ELSCAN).

    Raises
    ------
    ValueError
        If the AXIS-SAMPLE data has no packets or no samples inside the event window.
    """
    trimmed = slice_l1a_dataset_to_time_window(axis_sample, t0, t1)

    empty_dims = [dim for dim in ("PACKET", _AXIS_SAMPLE_TIME) if trimmed.sizes.get(dim, 0) == 0]
    if empty_dims:
        raise ValueError(
            f"AXIS-SAMPLE data has no data on {', '.join(empty_dims)} inside NOM-HK event window "
            f"[{t0} — {t1}]; cannot generate Azimuth/Elevation kernels"
        )

    sample_times = trimmed[_AXIS_SAMPLE_TIME].values
    logger.info(
        "AXIS-SAMPLE: %d / %d packets selected, %d / %d %s samples, covering [%s — %s]",
        trimmed.sizes.get("PACKET", 0),
        axis_sample.sizes.get("PACKET", 0),
        trimmed.sizes[_AXIS_SAMPLE_TIME],
        axis_sample.sizes.get(_AXIS_SAMPLE_TIME, 0),
        _AXIS_SAMPLE_TIME,
        sample_times.min(),
        sample_times.max(),
    )

    # A short base path keeps the generated kernel clear of SPICE's 80-character path limit,
    # matching the convention libera_utils' own make_kernel uses.
    with tempfile.TemporaryDirectory(prefix="/tmp/") as kernel_dir:  # nosec B108
        kernel_paths = [
            create_kernel_from_l1a(trimmed, product_id, kernel_dir, overwrite=True)
            for product_id in _GENERATED_SPICE_CAL_AZEL
        ]
        logger.info("Generated event kernels: %s", [Path(str(path)).name for path in kernel_paths])
        yield kernel_paths


def attach_azimuth_elevation_from_axis_sample(
    cal_event: xr.Dataset,
    axis_sample: xr.Dataset,
    t0: np.datetime64,
    t1: np.datetime64,
) -> xr.Dataset:
    """Generate this event's motor CKs from AXIS-SAMPLE and attach Azimuth/Elevation.

    The NOM-HK window defines the event, but a kernel has to *bracket* the times it is asked to
    interpolate, so the axis data is cut to that window widened to the samples actually queried.
    The two are not the same: companions are trimmed a whole packet at a time, and RAD and AXIS
    are sampled on independently clocked 200 Hz grids, so the RAD edge sample can land past the
    last AXIS sample of the same window — 10 ms past it, for 3 of 16150 samples, in ground-test
    solar data. Cutting the axis data to the bare window leaves those samples uncovered, and
    ``calculate_azimuth_elevation_for_timestamps`` turns an uncovered sample into ``-999.0``
    fill rather than an error. Widening by the queried range costs one extra axis packet per
    edge and removes that failure mode.

    Kernel creation completes before the query context opens: ``create_kernel_from_l1a`` furnishes
    kernels through its own ``KernelManager``, and the querying ``KernelManager`` clears the SPICE
    pool when it exits, so the two must not be interleaved.

    Parameters
    ----------
    cal_event : xr.Dataset
        Merged SWC/LWC/SOLAR calibration event dataset.
    axis_sample : xr.Dataset
        Decoded AXIS-SAMPLE-DECODED dataset for the day containing the event.
    t0 : np.datetime64
        Event window start (inclusive), from NOM-HK.
    t1 : np.datetime64
        Event window end (inclusive), from NOM-HK.

    Returns
    -------
    xr.Dataset
        ``cal_event`` with ``Azimuth_Position`` and ``Elevation_Position`` assigned.

    Raises
    ------
    ValueError
        If ``RAD_SAMPLE_FPE_TIME`` is missing from ``cal_event``.
    """
    if _RAD_SAMPLE_FPE_TIME not in cal_event:
        raise ValueError(f"Calibration event dataset is missing {_RAD_SAMPLE_FPE_TIME}")

    queried = cal_event[_RAD_SAMPLE_FPE_TIME].values
    kernel_t0 = min(t0, np.datetime64(queried.min()))
    kernel_t1 = max(t1, np.datetime64(queried.max()))
    if (kernel_t0, kernel_t1) != (t0, t1):
        logger.info(
            "Widening kernel window from NOM-HK [%s — %s] to [%s — %s] to cover the queried %s samples",
            t0,
            t1,
            kernel_t0,
            kernel_t1,
            _RAD_SAMPLE_FPE_TIME,
        )

    with event_azel_kernels(axis_sample, kernel_t0, kernel_t1) as kernel_paths:
        return attach_azimuth_elevation_positions(cal_event, kernel_paths)


def family_needs_azimuth_elevation_positions(family: DataProductIdentifier) -> bool:
    """Return True when the calibration family writes Azimuth/Elevation positions.

    Parameters
    ----------
    family : DataProductIdentifier
        Family TRIMMED ProductID (``CalEventSpec.trimmed_product``).
    """
    return family in _AZEL_POSITION_FAMILIES


def extract_named_nom_hk_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> tuple[str, xr.Dataset]:
    """Return the ``(filename, dataset)`` of the event's family TRIMMED NOM-HK granule.

    The trimmed granule is what defines the calibration event: it carries one ObsID, and its
    packet time range is the window every companion is cropped to.

    Exactly one granule must be present. A family TRIMMED ProductID names a whole calibration
    family rather than one file: ``nom_hk_trim`` writes one per contiguous ObsID run.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    tuple of (str, xr.Dataset)
        Source filename and TRIMMED NOM-HK dataset for the event.

    Raises
    ------
    ValueError
        If the family TRIMMED NOM-HK product is absent from the inputs, or if more than one of
        its granules is present.
    """
    product_id = event_spec.trimmed_product
    matches = [
        (file_name, dataset)
        for file_name, dataset in all_data.items()
        if LiberaDataProductFilename.from_file_path(file_name).data_product_id == product_id.value
    ]
    if not matches:
        raise ValueError(
            f"No {product_id.value} granule in the input files; cal-combine requires the "
            f"trimmed NOM-HK granule that defines the calibration event"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Input manifest carries {len(matches)} {product_id.value} granules; cal-combine "
            f"expects one NOM-HK granule per calibration event: "
            f"{', '.join(Path(str(name)).name for name, _ in matches)}"
        )
    file_name, dataset = matches[0]
    logger.info("Using NOM-HK input product %s from %s", product_id.value, file_name)
    return file_name, dataset


def extract_nom_hk_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Return the event's family TRIMMED NOM-HK dataset.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    xr.Dataset
        NOM-HK dataset for the event.
    """
    return extract_named_nom_hk_dataset(all_data, event_spec)[1]


def confirm_obsid_matches_hk(nom_hk: xr.Dataset, expected_obsid: int) -> None:
    """Fail-closed confirmation that HK ObsIDs match ``LIBERA_CAL_OBSID``.

    Parameters
    ----------
    nom_hk : xr.Dataset
        TRIMMED NOM-HK dataset containing ``ICIE__SW_OBSID_RAD``.
    expected_obsid : int
        ObsID from ``LIBERA_CAL_OBSID``.

    Raises
    ------
    ValueError
        If the HK packets do not consistently match ``expected_obsid``.
    """
    if "ICIE__SW_OBSID_RAD" not in nom_hk:
        raise ValueError("NOM-HK dataset is missing ICIE__SW_OBSID_RAD required for ObsID confirmation")
    obsids = np.unique(nom_hk["ICIE__SW_OBSID_RAD"].values)
    known_cal = set(CAL_EVENT_BY_OBSID)
    present_cal = sorted(int(o) for o in obsids if int(o) in known_cal)
    if expected_obsid not in present_cal:
        raise ValueError(
            f"ObsID confirmation failed: {LIBERA_CAL_OBSID_ENV}={expected_obsid} not found in "
            f"NOM-HK ICIE__SW_OBSID_RAD values {obsids.tolist()}"
        )
    other_cal = [o for o in present_cal if o != expected_obsid]
    if other_cal:
        raise ValueError(
            f"ObsID confirmation failed: NOM-HK contains additional calibration ObsIDs {other_cal}; "
            f"expected only {expected_obsid}"
        )
    logger.info("Confirmed NOM-HK ObsIDs are consistent with %s=%d", LIBERA_CAL_OBSID_ENV, expected_obsid)


def nom_hk_event_window(nom_hk: xr.Dataset) -> tuple[np.datetime64, np.datetime64]:
    """Return inclusive ``(t0, t1)`` bounds from NOM-HK ``PACKET_ICIE_TIME``.

    Parameters
    ----------
    nom_hk : xr.Dataset
        NOM-HK dataset for the calibration event.

    Returns
    -------
    tuple of np.datetime64
        Inclusive event window start and end.

    Raises
    ------
    ValueError
        If ``PACKET_ICIE_TIME`` is missing or empty.
    """
    if "PACKET_ICIE_TIME" not in nom_hk:
        raise ValueError("NOM-HK dataset is missing PACKET_ICIE_TIME required for event-window slicing")
    times = nom_hk["PACKET_ICIE_TIME"].values
    if times.size == 0:
        raise ValueError("NOM-HK dataset has no packets; cannot derive an event window")
    return np.datetime64(times.min()), np.datetime64(times.max())


def _extract_named_input_dataset(
    all_data: dict[str, xr.Dataset], product_id: DataProductIdentifier
) -> tuple[str, xr.Dataset]:
    """Return the ``(filename, dataset)`` matching ``product_id`` from the loaded inputs."""
    for file_name in all_data:
        if LiberaDataProductFilename.from_file_path(file_name).data_product_id == product_id.value:
            return file_name, all_data[file_name]
    raise ValueError("No dataset found in input files: " + product_id.value)


def extract_kernel_source_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> tuple[str, xr.Dataset]:
    """Return the ``(filename, dataset)`` of the event's SPICE kernel source granule.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    tuple of (str, xr.Dataset)
        Source filename and kernel source dataset (AXIS-SAMPLE-DECODED).

    Raises
    ------
    ValueError
        If the family declares no kernel source, declares more than one, or the granule is
        missing from the inputs.
    """
    kernel_sources = event_spec.kernel_source_products
    if len(kernel_sources) != 1:
        raise ValueError(
            f"Calibration family {event_spec.trimmed_product.value} declares {len(kernel_sources)} kernel "
            f"source products ({[p.value for p in kernel_sources]}); exactly one is required to generate "
            f"Azimuth/Elevation kernels"
        )
    return _extract_named_input_dataset(all_data, kernel_sources[0])


def select_and_slice_event_inputs(
    all_data: dict[str, xr.Dataset],
    event_spec: CalEventSpec,
) -> tuple[list[xr.Dataset], list[str]]:
    """Select NOM-HK plus companions and slice companions to the event window.

    Companions are sliced by
    :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window`, which selects
    whole packets on *sample* time where sample axes exist and renumbers each
    ``*_packet_index`` from 0 against the surviving packet axis.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    list of xr.Dataset
        ``[nom_hk, *sliced_companions]`` suitable for
        :func:`~libera_rad.calibration.combiners.l1a_combine.merge_l1a_decoded_datasets`.
    list of str
        Filenames of the L1A inputs used, NOM-HK first, for the product's ``input_files``.

    Raises
    ------
    ValueError
        If NOM-HK or any required companion product is missing, or if the
        NOM-HK window cannot be derived.
    """
    nom_hk_file_name, nom_hk = extract_named_nom_hk_dataset(all_data, event_spec)

    t0, t1 = nom_hk_event_window(nom_hk)
    logger.info("Slicing companions to event window: [%s — %s]", t0, t1)

    sliced_companions: list[xr.Dataset] = []
    input_file_names: list[str] = [nom_hk_file_name]
    for product_id in event_spec.companion_products:
        companion_file_name, companion = _extract_named_input_dataset(all_data, product_id)
        sliced = slice_l1a_dataset_to_time_window(companion, t0, t1)
        sample_dims = sorted(find_sample_dims(companion))
        logger.info(
            "%s: %d / %d packets selected%s",
            product_id.value,
            sliced.sizes.get("PACKET", 0),
            companion.sizes.get("PACKET", 0),
            "".join(f", {sliced.sizes[dim]} / {companion.sizes[dim]} {dim} samples" for dim in sample_dims),
        )
        empty_dims = [dim for dim in ("PACKET", *sample_dims) if sliced.sizes.get(dim, 0) == 0]
        if empty_dims:
            raise ValueError(
                f"Companion product {product_id.value} has no data on {', '.join(empty_dims)} inside "
                f"NOM-HK event window [{t0} — {t1}]"
            )
        sliced_companions.append(sliced)
        input_file_names.append(companion_file_name)

    return [nom_hk, *sliced_companions], input_file_names


def _merge_event_streams(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Slice companions to the NOM-HK window and merge streams."""
    inputs, input_file_names = select_and_slice_event_inputs(all_data, event_spec)
    merged = merge_l1a_decoded_datasets(inputs)
    merged.attrs[_INPUT_FILES_ATTR] = [Path(str(name)).name for name in input_file_names]
    return merged


def build_event_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Build one ObsID calibration event dataset for ``event_spec.trimmed_product``.

    Every family uses the same select → slice → merge path.

    Sets ``source_obsids``, ``algorithm_version`` and ``date_created`` so every family writes a
    uniform product header. ``date_created`` must be set here because the merge inherits its
    attributes from the NOM-HK input, whose timestamp is the Step-1 trim's, not this run's.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    xr.Dataset
        Merged calibration event dataset.
    """
    logger.info("Creating %s calibration event dataset (ObsID %d)", event_spec.trimmed_product.value, event_spec.obsid)
    event = _merge_event_streams(all_data, event_spec)
    event.attrs["source_obsids"] = [event_spec.obsid]
    event.attrs["algorithm_version"] = libera_rad_version()
    event.attrs["date_created"] = datetime.now(tz=UTC).isoformat()
    return event


def add_input_files(cal_event: xr.Dataset, file_names: Sequence[str | Path | S3Path]) -> xr.Dataset:
    """Append granule names to the product's ``input_files`` attribute, de-duplicated.

    Used for inputs that are only known outside the event builder, such as the SPICE kernels
    furnished for Azimuth/Elevation.

    Parameters
    ----------
    cal_event : xr.Dataset
        Calibration event dataset.
    file_names : sequence of str, Path, or S3Path
        Paths to record; only the basename is stored.

    Returns
    -------
    xr.Dataset
        ``cal_event`` with its ``input_files`` attribute extended.
    """
    existing = list(cal_event.attrs.get(_INPUT_FILES_ATTR, []))
    for name in file_names:
        base_name = Path(str(name)).name
        if base_name not in existing:
            existing.append(base_name)
    cal_event.attrs[_INPUT_FILES_ATTR] = existing
    return cal_event
