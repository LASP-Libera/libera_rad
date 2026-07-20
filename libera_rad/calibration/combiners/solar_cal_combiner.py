"""Solar calibration event family merge (one ObsID per product)."""

from __future__ import annotations

import logging

import xarray as xr

from libera_rad.calibration.combiners import l1a_cal_event_utils, l1a_combine
from libera_rad.calibration.combiners.l1a_cal_event_utils import extract_nom_hk_dataset
from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    SOLAR_FACE_BASE_OBSIDS,
    SOLAR_OBSID_TO_FACE_NUM,
    CalEventSpec,
)
from libera_rad.version import version as libera_rad_version

logger = logging.getLogger(__name__)


def build_event_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Build a single-ObsID solar calibration event dataset.

    Filters NOM-HK to ``event_spec.obsid``, slices companion products to that
    time window, merges streams, and sets solar-cal global attributes.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific solar calibration event specification.

    Returns
    -------
    xr.Dataset
        Merged solar calibration event dataset.
    """
    nom_hk = extract_nom_hk_dataset(all_data, event_spec)
    obsid_mask = nom_hk["ICIE__SW_OBSID_RAD"].values == event_spec.obsid
    nom_hk_event = nom_hk.isel(PACKET=obsid_mask)
    logger.info(
        "NOM-HK: %d / %d packets retained for ObsID %d",
        int(obsid_mask.sum()),
        len(obsid_mask),
        event_spec.obsid,
    )
    if nom_hk_event.sizes.get("PACKET", 0) == 0:
        raise ValueError(f"No NOM-HK packets found for solar-cal ObsID {event_spec.obsid}")

    inputs = l1a_cal_event_utils.select_and_slice_event_inputs(all_data, event_spec, nom_hk=nom_hk_event)
    solar_cal_event = l1a_combine.merge_l1a_decoded_datasets(inputs)

    face_num = SOLAR_OBSID_TO_FACE_NUM[event_spec.obsid]
    event_pass_index = event_spec.obsid - SOLAR_FACE_BASE_OBSIDS[face_num]
    solar_cal_event.attrs["solar_cal_face"] = face_num
    solar_cal_event.attrs["source_obsids"] = [event_spec.obsid]
    solar_cal_event.attrs["event_pass_index"] = event_pass_index
    solar_cal_event.attrs["algorithm_version"] = libera_rad_version()
    logger.info(
        "Global attributes set: solar_cal_face=%d, source_obsids=%s, event_pass_index=%d",
        face_num,
        [event_spec.obsid],
        event_pass_index,
    )
    return solar_cal_event


def solar_obsids() -> set[int]:
    """Return the set of known solar-cal ObsIDs."""
    return {obsid for obsid, spec in CAL_EVENT_BY_OBSID.items() if spec.family == "solar"}
