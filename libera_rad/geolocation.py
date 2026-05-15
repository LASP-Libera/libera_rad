"""
Geolocation Calculations.

This module provides a clean interface for managing SPICE kernels and performing
geolocation calculations for the Libera instrument. It separates kernel lifecycle
management from computation logic to improve maintainability and performance.
"""

import logging

import numpy as np
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

    u_gps_times = spicetime.adapt(time_range, "iso")

    ellips_lla_df, sc_xyz_df, ellips_qf_ds = spatial.compute_ellipsoid_intersection(
        u_gps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), give_geodetic_output=True, give_lat_lon_in_degrees=True
    )

    logger.debug(f"Calculation complete, generated {len(ellips_lla_df)} points")

    return ellips_lla_df


def calculate_geolocation_for_timestamps(km: KernelManager, timestamps: np.ndarray) -> pd.DataFrame:
    """
    Calculate geolocation information for given timestamps using SPICE kernels.

    Parameters
    ----------
    km : KernelManager
        Initialized kernel manager with loaded SPICE kernels.
    timestamps : np.ndarray
        Array of timestamps (datetime64 or float) for which to calculate positions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing geolocation data.
    """
    # TODO[LIBSDC-739]: Add all geolocation values into the data product
    datetime_index_time = pd.DatetimeIndex(timestamps)
    return calculate_lat_lon_altitude(km, datetime_index_time)
