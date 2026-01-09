"""
Geolocation Calculations.

This module provides a clean interface for managing SPICE kernels and performing
geolocation calculations for the Libera instrument. It separates kernel lifecycle
management from computation logic to improve maintainability and performance.
"""

import logging

import pandas as pd
from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import spatial
from libera_utils.libera_spice.kernel_manager import KernelManager

logger = logging.getLogger(__name__)


def calculate_lat_lon_altitude(
    kernel_manager: KernelManager,
    time_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate latitude, longitude, and altitude (convenience function).

    This function handles all kernel loading and cleanup automatically.
    For multiple calculations, use KernelManager directly for better performance.

    Parameters
    ----------
    kernel_manager: KernelManager
        An instance of KernelManager with kernels already loaded.
    time_range: pd.DatetimeIndex
        A range of times for which to calculate geolocation data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing latitude, longitude, and altitude data.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    # Could use more error handling here to support more options of inputs for time range
    u_gps_times = spicetime.adapt(time_range, "iso")

    ellips_lla_df, sc_xyz_df, ellips_qf_ds = spatial.instrument_intersect_ellipsoid(
        u_gps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), geodetic=True, degrees=True
    )

    logger.debug(f"Calculation complete, generated {len(ellips_lla_df)} points")

    return ellips_lla_df
