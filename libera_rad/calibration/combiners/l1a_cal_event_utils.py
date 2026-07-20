"""L1A calibration event utilities — manifest load and event-window slicing.

Shared helpers for ObsID-dispatched calibration combiners:

- Load decoded L1A datasets from a calibration input manifest
- Prefer TRIMMED NOM-HK inputs with legacy decoded fallback
- Confirm NOM-HK ObsIDs against ``LIBERA_CAL_OBSID``
- Select companion products and slice them to the NOM-HK event window
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from libera_utils import Manifest
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename

from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID, LIBERA_CAL_OBSID_ENV, CalEventSpec
from libera_rad.l1b import extract_input_dataset, read_all_input_data

logger = logging.getLogger(__name__)

#: Secondary FPE time dimension to slice for products that have one.
_COMPANION_SECONDARY_TIME_DIM: dict[DataProductIdentifier, str] = {
    DataProductIdentifier.l1a_icie_rad_sample_decoded: "RAD_SAMPLE_FPE_TIME",
    DataProductIdentifier.l1a_icie_cal_sample_decoded: "CAL_SAMPLE_FPE_TIME",
    DataProductIdentifier.l1a_icie_rad_full_decoded: "RAD_FULL_FPE_TIME",
    DataProductIdentifier.l1a_icie_cal_full_decoded: "CAL_FULL_FPE_TIME",
}


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


def read_calibration_manifest_data(input_manifest: Manifest) -> dict[str, xr.Dataset]:
    """Load decoded L1A datasets from a calibration combiner input manifest.

    Calibration combiners do not use SPICE/geolocation. Input manifests therefore
    default to ``use_geo: false`` so :func:`~libera_rad.l1b.read_all_input_data`
    does not require SPICE kernel products.
    """
    if input_manifest.configuration.get("use_geo", True):
        read_manifest = input_manifest.model_copy(
            update={"configuration": {**input_manifest.configuration, "use_geo": False}}
        )
    else:
        read_manifest = input_manifest
    all_data, _ = read_all_input_data(read_manifest)
    return all_data


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
