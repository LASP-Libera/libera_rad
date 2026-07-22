"""L1A calibration event utilities — manifest load, slicing, and event builders.

Shared helpers for ObsID-dispatched calibration products:

- Load decoded L1A datasets from a calibration input manifest
- Prefer TRIMMED NOM-HK inputs with legacy decoded fallback
- Confirm NOM-HK ObsIDs against ``LIBERA_CAL_OBSID``
- Select companion products and slice them to the NOM-HK event window
- Build family event datasets (default merge path + family-specific overrides)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import S3Path
from libera_utils import Manifest, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad import geolocation
from libera_rad.calibration.combiners.l1a_combine import merge_l1a_decoded_datasets
from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    LIBERA_CAL_OBSID_ENV,
    CalEventSpec,
    CalFamily,
)
from libera_rad.l1b import extract_input_dataset
from libera_rad.version import version as libera_rad_version

logger = logging.getLogger(__name__)

#: Motor CK products required for SWC/LWC/SOLAR Azimuth/Elevation (furnish order).
_REQUIRED_SPICE_CAL_AZEL: tuple[DataProductIdentifier, ...] = (
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_el_ck,
)

#: Secondary FPE time dimension to slice for products that have one.
_COMPANION_SECONDARY_TIME_DIM: dict[DataProductIdentifier, str] = {
    DataProductIdentifier.l1a_icie_rad_sample_decoded: "RAD_SAMPLE_FPE_TIME",
    DataProductIdentifier.l1a_icie_cal_sample_decoded: "CAL_SAMPLE_FPE_TIME",
    DataProductIdentifier.l1a_icie_rad_full_decoded: "RAD_FULL_FPE_TIME",
    DataProductIdentifier.l1a_icie_cal_full_decoded: "CAL_FULL_FPE_TIME",
}

#: Families that receive SPICE-derived Azimuth_Position / Elevation_Position.
_AZEL_POSITION_FAMILIES = frozenset({"swc", "lwc", "solar"})
_RAD_SAMPLE_FPE_TIME = "RAD_SAMPLE_FPE_TIME"
_AZIMUTH_POSITION = "Azimuth_Position"
_ELEVATION_POSITION = "Elevation_Position"


def slice_dataset_to_time_window(
    ds: xr.Dataset,
    t0: np.datetime64,
    t1: np.datetime64,
    packet_time_var: str = "PACKET_ICIE_TIME",
    secondary_time_dim: str | None = None,
) -> xr.Dataset:
    """Slice a decoded L1A Dataset to packets (and optionally samples) in ``[t0, t1]``.

    Applies an inclusive mask on *packet_time_var* along ``PACKET``. When
    *secondary_time_dim* is set, that dimension is sliced independently (e.g.
    ``RAD_SAMPLE_FPE_TIME`` on RAD-SAMPLE products).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``PACKET`` dimension indexed by *packet_time_var*.
    t0 : np.datetime64
        Window start time (inclusive).
    t1 : np.datetime64
        Window end time (inclusive).
    packet_time_var : str
        Name of the packet-level time coordinate (default ``"PACKET_ICIE_TIME"``).
    secondary_time_dim : str or None
        Name of an independent secondary time dimension to also slice (e.g.
        ``"RAD_SAMPLE_FPE_TIME"``).  When provided the Dataset is sliced along
        both ``PACKET`` and this dimension independently.

    Returns
    -------
    xr.Dataset
        Time-sliced dataset.
    """
    pkt_times = ds[packet_time_var].values
    pkt_mask = (pkt_times >= t0) & (pkt_times <= t1)
    sel: dict[str, np.ndarray] = {"PACKET": pkt_mask}

    if secondary_time_dim is not None and secondary_time_dim in ds.dims:
        sec_times = ds[secondary_time_dim].values
        sec_mask = (sec_times >= t0) & (sec_times <= t1)
        sel[secondary_time_dim] = sec_mask
        logger.debug(
            "slice_dataset_to_time_window [%s — %s]: %d packets, %d %s samples.",
            t0,
            t1,
            int(pkt_mask.sum()),
            int(sec_mask.sum()),
            secondary_time_dim,
        )
    else:
        logger.debug(
            "slice_dataset_to_time_window [%s — %s]: %d packets.",
            t0,
            t1,
            int(pkt_mask.sum()),
        )

    return ds.isel(sel)


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


def family_needs_azimuth_elevation_positions(family: str) -> bool:
    """Return True when the calibration family writes Azimuth/Elevation positions."""
    return family in _AZEL_POSITION_FAMILIES


def extract_nom_hk_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Return the NOM-HK dataset from trimmed or full-day decoded inputs.

    Prefers the event's TRIMMED product, then falls back to ``NOM-HK-DECODED``
    for fixtures that predate Step-1 trimming.

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

    Raises
    ------
    ValueError
        If neither a TRIMMED nor decoded NOM-HK product is present.
    """
    preferred = (event_spec.trimmed_product, DataProductIdentifier.l1a_icie_nom_hk_decoded)
    for product_id in preferred:
        for file_name, dataset in all_data.items():
            libera_filename = LiberaDataProductFilename.from_file_path(file_name)
            if libera_filename.data_product_id == product_id.value:
                logger.info("Using NOM-HK input product %s from %s", product_id.value, file_name)
                return dataset
    raise ValueError(
        "No NOM-HK dataset found in input files. Expected one of: " + ", ".join(p.value for p in preferred)
    )


def confirm_obsid_matches_hk(nom_hk: xr.Dataset, expected_obsid: int) -> None:
    """Fail-closed confirmation that HK ObsIDs match ``LIBERA_CAL_OBSID``.

    Parameters
    ----------
    nom_hk : xr.Dataset
        NOM-HK (trimmed or decoded) dataset containing ``ICIE__SW_OBSID_RAD``.
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


def select_and_slice_event_inputs(
    all_data: dict[str, xr.Dataset],
    event_spec: CalEventSpec,
    *,
    nom_hk: xr.Dataset | None = None,
) -> list[xr.Dataset]:
    """Select NOM-HK plus companions and slice companions to the event window.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.
    nom_hk : xr.Dataset or None
        Optional pre-selected NOM-HK dataset (e.g. already ObsID-filtered for
        solar). When omitted, the TRIMMED/decoded NOM-HK product is extracted.

    Returns
    -------
    list of xr.Dataset
        ``[nom_hk, *sliced_companions]`` suitable for
        :func:`~libera_rad.calibration.combiners.l1a_combine.merge_l1a_decoded_datasets`.

    Raises
    ------
    ValueError
        If NOM-HK or any required companion product is missing, or if the
        NOM-HK window cannot be derived.
    """
    if nom_hk is None:
        nom_hk = extract_nom_hk_dataset(all_data, event_spec)

    t0, t1 = nom_hk_event_window(nom_hk)
    logger.info("Slicing companions to event window: [%s — %s]", t0, t1)

    sliced_companions: list[xr.Dataset] = []
    for product_id in event_spec.companion_products:
        companion = extract_input_dataset(all_data, product_id)
        secondary = _COMPANION_SECONDARY_TIME_DIM.get(product_id)
        sliced = slice_dataset_to_time_window(companion, t0, t1, secondary_time_dim=secondary)
        logger.info(
            "%s: %d / %d packets selected%s",
            product_id.value,
            sliced.sizes.get("PACKET", 0),
            companion.sizes.get("PACKET", 0),
            f", {sliced.sizes[secondary]} / {companion.sizes[secondary]} {secondary} samples"
            if secondary and secondary in companion.dims
            else "",
        )
        if sliced.sizes.get("PACKET", 0) == 0:
            raise ValueError(
                f"Companion product {product_id.value} has no packets inside NOM-HK event window [{t0} — {t1}]"
            )
        sliced_companions.append(sliced)

    return [nom_hk, *sliced_companions]


EventBuilder = Callable[[dict[str, xr.Dataset], CalEventSpec], xr.Dataset]


def _build_standard_event_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Slice companions to the NOM-HK window and merge streams."""
    inputs = select_and_slice_event_inputs(all_data, event_spec)
    return merge_l1a_decoded_datasets(inputs)


#: Optional family-specific builders. Empty by default; all current families use the
#: standard path. Register an override when a future family needs custom prep.
_FAMILY_EVENT_BUILDER_OVERRIDES: dict[CalFamily, EventBuilder] = {}


def register_family_event_builder(family: CalFamily, builder: EventBuilder) -> None:
    """Register or replace the event builder for a calibration family.

    Parameters
    ----------
    family : CalFamily
        Calibration family key (must already be a valid ``CalEventSpec.family``).
    builder : callable
        ``(all_data, event_spec) -> xr.Dataset`` implementation.
    """
    _FAMILY_EVENT_BUILDER_OVERRIDES[family] = builder


def build_event_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Build one ObsID calibration event dataset for ``event_spec.family``.

    All current families use the same select → slice → merge path. Families with
    extra prep can be registered via :func:`register_family_event_builder`.

    Common global attributes (``source_obsids``, ``algorithm_version``) are set
    here so all cal families write a uniform product header.

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
    builder = _FAMILY_EVENT_BUILDER_OVERRIDES.get(event_spec.family, _build_standard_event_dataset)
    logger.info("Creating %s calibration event dataset (ObsID %d)", event_spec.family, event_spec.obsid)
    event = builder(all_data, event_spec)
    event.attrs["source_obsids"] = [event_spec.obsid]
    event.attrs["algorithm_version"] = libera_rad_version()
    return event
