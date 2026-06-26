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
from curryer.compute import geometry, spatial
from libera_utils.libera_spice.kernel_manager import KernelManager
from spiceypy.utils.exceptions import SpiceyError

logger = logging.getLogger(__name__)


def calculate_lat_lon_altitude(
    kernel_manager: KernelManager,
    time_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate instrument geolocation (latitude/longitude/altitude).

    Instrument lat/lon/alt come from the ``LIBERA_SW_RAD`` boresight ellipsoid
    intersection. The subsatellite point and the other ancillary geometry fields come
    from :func:`calculate_geometry_ancillary`.

    Parameters
    ----------
    kernel_manager: KernelManager
        An instance of KernelManager with kernels already loaded.
    time_range: pd.DatetimeIndex
        A range of times for which to calculate geolocation data.

    Returns
    -------
    pd.DataFrame
        Instrument geolocation with columns ``lat``, ``lon``, and ``alt``.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    u_gps_times = spicetime.adapt(time_range, "iso")

    ellips_lla_df, _, _ = spatial.compute_ellipsoid_intersection(
        u_gps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), give_geodetic_output=True, give_lat_lon_in_degrees=True
    )

    logger.debug("Instrument geolocation: generated %d points", len(ellips_lla_df))

    return ellips_lla_df


def calculate_geolocation_for_timestamps(km: KernelManager, timestamps: np.ndarray) -> pd.DataFrame:
    """
    Calculate instrument geolocation for given timestamps.

    Parameters
    ----------
    km : KernelManager
        Initialized kernel manager with loaded SPICE kernels.
    timestamps : np.ndarray
        Array of timestamps (datetime64 or float) for which to calculate positions.

    Returns
    -------
    pd.DataFrame
        Instrument geolocation DataFrame with columns ``lat``, ``lon``, ``alt``.
    """
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


def calculate_geometry_ancillary(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    observer: str = "JPSS4_SC",
) -> pd.DataFrame:
    """
    Compute position-derived geometry ancillary fields via curryer ``GeometryData``.

    curryer's selective-compute registry queries each SPICE input once, vectorized,
    with coverage gaps surfaced as NaN. One call yields the subsatellite and subsolar
    points (latitude / longitude / colatitude) plus the satellite radius and altitude
    -- product ancillary fields otherwise filled with -999.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the spacecraft SPK (and required FK) kernels furnished.
    timestamps : np.ndarray
        Radiometer timestamps on the L1B output time grid.
    observer : str
        SPICE body name for the spacecraft. Default ``"JPSS4_SC"``.

    Returns
    -------
    pd.DataFrame
        Indexed by uGPS; columns are the curryer field columns (subsatellite and
        subsolar latitude / longitude / colatitude, ``spacecraft_radius``,
        ``spacecraft_altitude``).
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    return geometry.GeometryData(observer).get_geometry(
        u_gps_times,
        fields=["subsatellite", "subsolar", "sc_radius", "sc_altitude"],
    )


def subsatellite_lat_lon_alt(geometry_ancillary: pd.DataFrame) -> pd.DataFrame:
    """
    Extract subsatellite ``lat``/``lon``/``alt`` from the ancillary fields.

    The subsatellite ground point's latitude/longitude plus the spacecraft geodetic
    altitude, in the columns the product packager consumes. Used as the geolocation in
    ``jpss_only`` mode, where no instrument pointing is available.

    Parameters
    ----------
    geometry_ancillary : pd.DataFrame
        Output of :func:`calculate_geometry_ancillary`.

    Returns
    -------
    pd.DataFrame
        Columns ``lat``, ``lon``, ``alt``.
    """
    return pd.DataFrame(
        {
            "lat": geometry_ancillary["subsatellite_latitude"].to_numpy(),
            "lon": geometry_ancillary["subsatellite_longitude"].to_numpy(),
            "alt": geometry_ancillary["spacecraft_altitude"].to_numpy(),
        }
    )


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


def _az_el_from_et(et: float) -> tuple[float, float] | None:
    """
    Motor encoder azimuth and elevation (degrees) at one ET epoch.

    Angles are Euler angles from the CK frame chain
    ``LIBERA_BASE_COORD → LIBERA_AZ_COORD → LIBERA_EL_COORD``, matching
    ``libera_utils`` tier-0 kernel tests. Returns None when SPICE has no transform.
    """
    # TODO[LIBSDC-739]: Use curryer frame_euler + spice_error_to_val when curryer#158 lands.
    try:
        # TODO[LIBSDC-611]: Re-evaluate the naming of the frames.
        _TWO_PI = float(2.0 * np.pi)
        m_az = sp.pxform("LIBERA_BASE_COORD", "LIBERA_AZ_COORD", et)
        az_rad = float(sp.m2eul(m_az, 1, 2, 3)[2])
        az_deg = np.degrees((az_rad + _TWO_PI) % _TWO_PI)

        m_el = sp.pxform("LIBERA_AZ_COORD", "LIBERA_EL_COORD", et)
        el_rad = float(sp.m2eul(m_el, 1, 2, 3)[0])
        el_deg = np.degrees(el_rad)
    except SpiceyError:
        logger.debug("SPICE frame transform unavailable at ET %.6f", et, exc_info=True)
        return None

    return az_deg, el_deg


def _az_el_on_et_times(et_times: np.ndarray, fill_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute az/el at every ET sample (full-resolution SPICE path)."""
    az = np.full(shape=len(et_times), fill_value=fill_value, dtype=np.float32)
    el = np.full(shape=len(et_times), fill_value=fill_value, dtype=np.float32)

    for i, et in enumerate(np.asarray(et_times, dtype=np.float64)):
        result = _az_el_from_et(float(et))
        if result is not None:
            az[i] = np.float32(result[0])
            el[i] = np.float32(result[1])

    return az, el


def calculate_azimuth_elevation_for_timestamps(
    km: KernelManager,
    timestamps: np.ndarray,
    fill_value: float = -999.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate instrument azimuth and elevation angles for the given timestamps.

    Angles are motor encoder Euler angles from SPICE CK frame transforms:
    azimuth from ``LIBERA_BASE_COORD → LIBERA_AZ_COORD`` and elevation from
    ``LIBERA_AZ_COORD → LIBERA_EL_COORD``, using conventions validated in
    ``libera_utils`` tier-0 kernel tests. They are relative to the motor encoder
    frames, not nadir or spacecraft attitude.

    Parameters
    ----------
    km : KernelManager
        Initialized kernel manager with loaded SPICE kernels.
    timestamps : np.ndarray
        Radiometer timestamps aligned to the L1B output time grid as ``datetime64[ns]``.
    fill_value : float
        Fill value for samples where SPICE frame transforms are unavailable (coverage gaps).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(azimuth_deg, elevation_deg)` arrays, each shape `(N,)`, dtype float32.
    """
    km.ensure_known_kernels_are_furnished()

    # TODO[LIBSDC-788]: CK coverage check (via KernelManager?) before az/el pxform loop.

    dt64_times = np.asarray(timestamps, dtype="datetime64[ns]")
    et_times = spicetime.adapt(dt64_times, "dt64", "et")
    return _az_el_on_et_times(et_times, fill_value)
