"""Gain calibration event family merge."""

import logging

import xarray as xr

from libera_rad.calibration.combiners import l1a_cal_event_utils, l1a_combine
from libera_rad.calibration.constants import CalEventSpec
from libera_rad.version import version as libera_rad_version

logger = logging.getLogger(__name__)


def build_event_dataset(all_data: dict[str, xr.Dataset], event_spec: CalEventSpec) -> xr.Dataset:
    """Merge gain-cal L1A inputs into one event-windowed dataset.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Decoded L1A datasets keyed by filename.
    event_spec : CalEventSpec
        ObsID-specific gain calibration event specification.

    Returns
    -------
    xr.Dataset
        Merged gain calibration event dataset.
    """
    logger.info("Creating gain calibration event dataset")
    inputs = l1a_cal_event_utils.select_and_slice_event_inputs(all_data, event_spec)
    gain_event = l1a_combine.merge_l1a_decoded_datasets(inputs)
    gain_event.attrs["algorithm_version"] = libera_rad_version()
    return gain_event
