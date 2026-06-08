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

# LIBERA_BASE_COORD +Z nadir; verified on integration JPSS-only kernels (no motor CK).
_LIBERA_BASE_NADIR_VECTOR = np.array([0.0, 0.0, 1.0], dtype=np.float64)


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
    return calculate_lat_lon_altitude(km, pd.DatetimeIndex(timestamps))


def create_placeholder_azimuth_elevation(n_samples: int, fill_value: float = -999.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder motor angles for no-geolocation mode.

    Parameters
    ----------
    n_samples : int
        Number of samples on the L1B output time grid.
    fill_value : float
        Product fill value for Azimuth and Elevation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(azimuth_deg, elevation_deg)` arrays, dtype float32.
    """
    fill = np.float32(fill_value)
    return (
        np.full(n_samples, fill, dtype=np.float32),
        np.full(n_samples, fill, dtype=np.float32),
    )


def create_jpss_only_motor_angles(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference motor angles for jpss_only mode (no motor CK).

    Returns 0° azimuth and elevation per operational convention when motor
    kernels are unavailable.
    """
    zeros = np.zeros(n_samples, dtype=np.float32)
    return zeros, zeros


def calculate_libera_base_subsatellite_geolocation(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Subsatellite geolocation using LIBERA_BASE nadir ellipsoid intersection.

    Uses JPSS dynamic kernels (SPK/CK) plus static FK; does not require motor
    azimuth/elevation CK. Surface LLA (altitude near 0 m on the ellipsoid).

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with JPSS and static kernels already loaded.
    timestamps : np.ndarray
        Radiometer timestamps on the L1B 100 Hz grid.

    Returns
    -------
    pd.DataFrame
        Columns ``lat``, ``lon``, ``alt`` in degrees / meters.
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")

    ellips_lla_df, _, _ = spatial.compute_ellipsoid_intersection(
        u_gps_times,
        sp.obj.Body("LIBERA_BASE", frame=True),
        custom_pointing_vectors=_LIBERA_BASE_NADIR_VECTOR,
        give_geodetic_output=True,
        give_lat_lon_in_degrees=True,
    )
    logger.debug("LIBERA_BASE subsatellite geolocation: %d points", len(ellips_lla_df))
    return ellips_lla_df


def create_placeholder_geolocation_dataframe(n_samples: int) -> pd.DataFrame:
    """
    Create placeholder geolocation values when use_geo is False.

    Used when SPICE geolocation is intentionally bypassed via manifest
    ``configuration.use_geo``. Returns latitude, longitude, and altitude values
    filled with standard product fill values.

    Parameters
    ----------
    n_samples : int
        Number of geolocation rows to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``lat``, ``lon``, and ``alt``.
    """
    logger.info("use_geo is false: using placeholder geolocation (Latitude, Longitude, Altitude).")
    return pd.DataFrame(
        {
            "lat": np.full(shape=n_samples, fill_value=-999, dtype=np.float32),
            "lon": np.full(shape=n_samples, fill_value=-999, dtype=np.float32),
            "alt": np.full(shape=n_samples, fill_value=-9999, dtype=np.float32),
        }
    )
