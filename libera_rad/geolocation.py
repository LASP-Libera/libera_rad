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


def _subsatellite_lla_from_ecef(sc_xyz_df: pd.DataFrame) -> pd.DataFrame:
    """Derive subsatellite geodetic coordinates from spacecraft ECEF position."""
    lla = spatial.ecef_to_geodetic(sc_xyz_df[["x", "y", "z"]].to_numpy(), meters=False, degrees=True)
    return pd.DataFrame({"lat": lla[..., 1], "lon": lla[..., 0], "alt": lla[..., 2]})


def _spacecraft_ecef_positions(u_gps_times: np.ndarray, spice_body_name: str = "JPSS4_SC") -> pd.DataFrame:
    """
    Query spacecraft ECEF positions at each time (no instrument kernel or pointing).

    Uses ``SpatialQueries.query_rotation_and_position`` for ``spkezp`` in ITRF93;
    only the position vector is retained. Gaps or kernel errors leave NaN rows.
    """
    instrument = sp.obj.Body(spice_body_name, frame=True)
    et_times = spicetime.adapt(u_gps_times, to="et")
    positions = np.full((len(et_times), 3), np.nan)

    for ith, et_time in enumerate(et_times):
        (_, position), _ = spatial.SpatialQueries.query_rotation_and_position(
            et_time, instrument, "NONE", allow_nans=True
        )
        if np.isfinite(position).all():
            positions[ith, :] = position

    index = pd.Index(np.asarray(u_gps_times).ravel(), name="ugps")
    return pd.DataFrame(positions, columns=["x", "y", "z"], index=index)


def calculate_lat_lon_altitude(
    kernel_manager: KernelManager,
    time_range: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate instrument and subsatellite geolocation (convenience function).

    This function handles all kernel loading and cleanup automatically.
    For multiple calculations, use KernelManager directly for better performance.

    Instrument lat/lon/alt come from ``LIBERA_SW_RAD`` ellipsoid intersection.
    Subsatellite lat/lon/alt come from spacecraft ECEF position via
    ``spatial.ecef_to_geodetic`` (attitude-independent).

    Parameters
    ----------
    kernel_manager: KernelManager
        An instance of KernelManager with kernels already loaded.
    time_range: pd.DatetimeIndex
        A range of times for which to calculate geolocation data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Instrument geolocation and subsatellite geolocation DataFrames with
        columns ``lat``, ``lon``, and ``alt``.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    u_gps_times = spicetime.adapt(time_range, "iso")

    ellips_lla_df, sc_xyz_df, _ = spatial.compute_ellipsoid_intersection(
        u_gps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), give_geodetic_output=True, give_lat_lon_in_degrees=True
    )
    subsatellite_lla_df = _subsatellite_lla_from_ecef(sc_xyz_df)

    logger.debug("Calculation complete, generated %d instrument and subsatellite points", len(ellips_lla_df))

    return ellips_lla_df, subsatellite_lla_df


def calculate_geolocation_for_timestamps(
    km: KernelManager, timestamps: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate instrument and subsatellite geolocation for given timestamps.

    Parameters
    ----------
    km : KernelManager
        Initialized kernel manager with loaded SPICE kernels.
    timestamps : np.ndarray
        Array of timestamps (datetime64 or float) for which to calculate positions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Instrument geolocation and subsatellite geolocation DataFrames.
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

    Parameters
    ----------
    n_samples : int
        Number of samples on the L1B output time grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(azimuth_deg, elevation_deg)` arrays, dtype float32, filled with zeros.
    """
    zeros = np.zeros(n_samples, dtype=np.float32)
    return zeros, zeros


def calculate_libera_base_subsatellite_geolocation(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Subsatellite geolocation from LIBERA_BASE spacecraft ECEF position.

    Uses JPSS dynamic kernels (SPK/CK) plus static FK; does not require motor
    azimuth/elevation CK or instrument pointing kernels. Spacecraft ECEF position
    is queried via SPICE ``spkezp``; subsatellite lat/lon come from
    ``spatial.ecef_to_geodetic``.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with JPSS and static kernels already loaded.
    timestamps : np.ndarray
        Radiometer timestamps on the L1B 100 Hz grid.

    Returns
    -------
    pd.DataFrame
        Columns ``lat``, ``lon``, ``alt`` in degrees / kilometers.
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")

    sc_xyz_df = _spacecraft_ecef_positions(u_gps_times)
    subsatellite_lla_df = _subsatellite_lla_from_ecef(sc_xyz_df)
    logger.debug("LIBERA_BASE subsatellite geolocation: %d points", len(subsatellite_lla_df))
    return subsatellite_lla_df


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
