"""L1A calibration event utilities — manifest load, slicing, and event building.

Shared helpers for ObsID-dispatched calibration products:

- Load decoded L1A datasets from a calibration input manifest
- Select the family TRIMMED NOM-HK granule that defines the event
- Confirm NOM-HK ObsIDs against ``LIBERA_CAL_OBSID``
- Select companion products and slice them to the NOM-HK event window
- Merge the selected streams into one calibration event dataset
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import S3Path
from libera_utils import Manifest, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
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

#: Motor CK products required for SWC/LWC/SOLAR Azimuth/Elevation (furnish order).
_REQUIRED_SPICE_CAL_AZEL: tuple[DataProductIdentifier, ...] = (
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_el_ck,
)

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


def read_all_cal_input_data(
    input_manifest: Manifest,
    *,
    require_azel_kernels: bool = True,
) -> tuple[dict[str, xr.Dataset], list[str]]:
    """Load decoded L1A datasets and optional AZROT/ELSCAN kernels from a cal manifest.

    When ``require_azel_kernels`` is true (SWC/LWC/SOLAR), AZROT-CK and ELSCAN-CK
    are required and returned in furnish order. Other SPICE products on the
    manifest are skipped with a warning.

    When ``require_azel_kernels`` is false (GAIN), SPICE products are never
    required; any ``.bc``/``.bsp`` files on the manifest are skipped.

    Parameters
    ----------
    input_manifest : Manifest
        Calibration combiner input manifest.
    require_azel_kernels : bool, optional
        Whether this calibration family needs motor AZROT/ELSCAN kernels.

    Returns
    -------
    dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    list of str
        Dynamic kernel paths in furnish order (AZROT then ELSCAN), or empty
        when kernels are not required.

    Raises
    ------
    ValueError
        If required AZROT/ELSCAN kernels are missing or duplicated.
    Exception
        If any NetCDF file cannot be opened or is invalid.
    """
    all_data: dict[str, xr.Dataset] = {}
    spice_files: dict[DataProductIdentifier, str] = {}
    spice_allowlist = set(_REQUIRED_SPICE_CAL_AZEL)

    for i, file_info in enumerate(input_manifest.files):
        logger.info("Reading file %d/%d: %s", i + 1, len(input_manifest.files), file_info.filename)
        try:
            if file_info.filename.endswith((".bc", ".bsp")):
                if not require_azel_kernels:
                    logger.warning("Skipping SPICE kernel %s (not required for this cal family)", file_info.filename)
                    continue

                product_id = LiberaDataProductFilename.from_file_path(file_info.filename).data_product_id
                if product_id not in spice_allowlist:
                    logger.warning(
                        "Skipping SPICE file %s (%s); not in required AZROT/ELSCAN set",
                        file_info.filename,
                        product_id,
                    )
                    continue

                if product_id in spice_files:
                    raise ValueError(
                        f"Duplicate SPICE data product {product_id} in manifest: "
                        f"{spice_files[product_id]} and {file_info.filename}"
                    )

                spice_files[product_id] = file_info.filename
                logger.info("Recorded SPICE kernel %s (%s)", file_info.filename, product_id)
            else:
                with smart_open(file_info.filename) as file_handle:
                    LiberaDataProductFilename.from_file_path(file_info.filename)
                    dataset = xr.open_dataset(file_handle, decode_times=True).load()
                    all_data[file_info.filename] = dataset
                    logger.info("Successfully loaded dataset: %s", file_info.filename)
        except Exception:
            logger.error("Failed to process file %s", file_info.filename, exc_info=True)
            raise

    dynamic_kernel_sources: list[str] = []
    if require_azel_kernels:
        missing = [product_id for product_id in _REQUIRED_SPICE_CAL_AZEL if product_id not in spice_files]
        if missing:
            labels = ", ".join(str(product_id) for product_id in missing)
            raise ValueError(f"Input manifest missing required SPICE data products: {labels}")
        dynamic_kernel_sources = [spice_files[product_id] for product_id in _REQUIRED_SPICE_CAL_AZEL]

    logger.info(
        "Successfully loaded %d datasets and %d SPICE kernel paths",
        len(all_data),
        len(dynamic_kernel_sources),
    )
    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data, dynamic_kernel_sources


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
        AZROT-CK and ELSCAN-CK paths from the input manifest.

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
