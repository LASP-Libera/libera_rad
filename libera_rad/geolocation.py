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


def create_placeholder_surface_geometry_angles(n_samples: int) -> dict[str, np.ndarray]:
    """
    Placeholder SZA, VZA, and RAA for no-geolocation mode.

    Returns product fill values when SPICE geolocation is disabled via
    ``configuration.use_geo``.
    """
    fill = np.float32(-999.0)
    return {
        "solar_zenith": np.full(n_samples, fill, dtype=np.float32),
        "viewing_zenith": np.full(n_samples, fill, dtype=np.float32),
        "relative_azimuth": np.full(n_samples, fill, dtype=np.float32),
    }


def calculate_surface_geometry_angles(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    lat_lon_alt: pd.DataFrame,
    spacecraft_body: str,
    fill_value: float = -999.0,
) -> dict[str, np.ndarray]:
    """
    Compute surface solar zenith, viewing zenith, and relative azimuth angles.

    Angles are evaluated at the Earth observation point given by ``lat_lon_alt``
    using curryer ``surface_angles`` (geodetic zenith convention). Relative
    azimuth is the satellite azimuth minus the solar azimuth, wrapped to
    ``[0, 360)`` degrees clockwise from north at the surface.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with SPICE kernels already loaded (including NAIF for the Sun).
    timestamps : np.ndarray
        Radiometer timestamps on the L1B 100 Hz grid.
    lat_lon_alt : pd.DataFrame
        Columns ``lat``, ``lon``, ``alt`` in degrees and meters at each timestamp.
    spacecraft_body : str
        NAIF body for the viewing direction (e.g. ``LIBERA_SW_RAD`` or ``LIBERA_BASE``).
    fill_value : float
        Product fill for samples where angles cannot be computed.

    Returns
    -------
    dict[str, np.ndarray]
        ``solar_zenith``, ``viewing_zenith``, and ``relative_azimuth`` arrays (float32).
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    n_samples = len(timestamps)
    fill = np.float32(fill_value)

    ugps_times = np.asarray(spicetime.adapt(pd.DatetimeIndex(timestamps), "iso"))
    lon = lat_lon_alt["lon"].to_numpy(dtype=np.float64)
    lat = lat_lon_alt["lat"].to_numpy(dtype=np.float64)
    alt_m = lat_lon_alt["alt"].to_numpy(dtype=np.float64)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(alt_m)

    sza = np.full(n_samples, fill, dtype=np.float32)
    vza = np.full(n_samples, fill, dtype=np.float32)
    raa = np.full(n_samples, fill, dtype=np.float32)
    if not np.any(valid):
        return {"solar_zenith": sza, "viewing_zenith": vza, "relative_azimuth": raa}

    lon_lat_alt_km = np.column_stack([lon[valid], lat[valid], alt_m[valid] / 1000.0])
    surface_xyz = spatial.geodetic_to_ecef(lon_lat_alt_km, meters=False, degrees=True)
    surface_positions = pd.DataFrame(
        surface_xyz,
        columns=["x", "y", "z"],
        index=pd.Index(ugps_times[valid], name="ugps"),
    )

    sun_angles = spatial.surface_angles(
        surface_positions,
        target_obj="SUN",
        degrees=True,
        geocentric=False,
        allow_nans=True,
    )
    sat_angles = spatial.surface_angles(
        surface_positions,
        target_obj=sp.obj.Body(spacecraft_body, frame=True),
        degrees=True,
        geocentric=False,
        allow_nans=True,
    )

    sza_valid = sun_angles["zenith"].to_numpy(dtype=np.float64)
    vza_valid = sat_angles["zenith"].to_numpy(dtype=np.float64)
    raa_valid = (sat_angles["azimuth"] - sun_angles["azimuth"]).to_numpy(dtype=np.float64) % 360.0

    sza[valid] = np.where(np.isfinite(sza_valid), sza_valid, np.nan).astype(np.float32)
    vza[valid] = np.where(np.isfinite(vza_valid), vza_valid, np.nan).astype(np.float32)
    raa[valid] = np.where(np.isfinite(raa_valid), raa_valid, np.nan).astype(np.float32)

    for arr in (sza, vza, raa):
        arr[~np.isfinite(arr)] = fill

    return {"solar_zenith": sza, "viewing_zenith": vza, "relative_azimuth": raa}


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


def calculate_azimuth_elevation_for_timestamps(
    km: KernelManager,
    timestamps: np.ndarray,
    fill_value: float = -999.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate instrument azimuth and elevation angles for the given timestamps.

    Angles are derived from SPICE CK frame transformations (motor kernels) using the same Euler angle
    conventions validated in `libera_utils` tier-0 kernel tests.

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

    # TODO[LIBSDC-788]: Pre-flight CK coverage check via KernelManager before az/el pxform loop.

    dt64_times = np.asarray(timestamps, dtype="datetime64[ns]")
    et_times = spicetime.adapt(dt64_times, "dt64", "et")
    az = np.full(shape=len(dt64_times), fill_value=fill_value, dtype=np.float32)
    el = np.full(shape=len(dt64_times), fill_value=fill_value, dtype=np.float32)

    two_pi = float(2.0 * np.pi)
    for i, et in enumerate(np.asarray(et_times, dtype=np.float64)):
        try:
            # Match libera_utils/tests/integration/test_tier0_kernel.py conventions
            m_az = sp.pxform("LIBERA_BASE_COORD", "LIBERA_AZ_COORD", float(et))
            az_rad = float(sp.m2eul(m_az, 1, 2, 3)[2])
            az_deg = np.degrees((az_rad + two_pi) % two_pi)

            m_el = sp.pxform("LIBERA_AZ_COORD", "LIBERA_EL_COORD", float(et))
            el_rad = float(sp.m2eul(m_el, 1, 2, 3)[0])
            el_deg = np.degrees(el_rad)

            az[i] = np.float32(az_deg)
            el[i] = np.float32(el_deg)
        except Exception:
            logger.debug(
                "SPICE frame transform unavailable at ET %.6f; leaving azimuth/elevation fill",
                et,
                exc_info=True,
            )
            continue

    return az, el
