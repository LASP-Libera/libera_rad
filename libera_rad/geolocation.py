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
from curryer.spicierpy.ext import spice_error_message
from libera_utils.libera_spice.kernel_manager import KernelManager
from spiceypy.utils.exceptions import SpiceyError

logger = logging.getLogger(__name__)


# curryer geometry fields for the L1B product, as GeometryField enum members (interchangeable
# with their string selectors; ``.columns`` gives the output keys). Split by observer: the
# spacecraft fields resolve in every mode; the instrument/boresight fields need the motor CK.
_SPACECRAFT_FIELDS = (
    geometry.GeometryField.SUBSATELLITE,
    geometry.GeometryField.SUBSOLAR,
    geometry.GeometryField.SC_RADIUS,
    geometry.GeometryField.SC_ALTITUDE,
    geometry.GeometryField.SC_POSITION_INERTIAL,
    geometry.GeometryField.SC_VELOCITY_INERTIAL,
    geometry.GeometryField.SATELLITE_ATTITUDE,
)
_INSTRUMENT_FIELDS = (
    geometry.GeometryField.BORESIGHT_INERTIAL,
    geometry.GeometryField.VIEWING_ZENITH,
    geometry.GeometryField.SOLAR_ZENITH,
    geometry.GeometryField.VIEWING_AZIMUTH,
    geometry.GeometryField.SOLAR_AZIMUTH,
    geometry.GeometryField.RELATIVE_AZIMUTH,
    geometry.GeometryField.CONE_ANGLE,
    geometry.GeometryField.CONE_ANGLE_RATE,
    geometry.GeometryField.CLOCK_ANGLE,
    geometry.GeometryField.CLOCK_ANGLE_RATE,
    geometry.GeometryField.ALONG_TRACK_ANGLE,
    geometry.GeometryField.CROSS_TRACK_ANGLE,
)
_GEOMETRY_FIELDS = _SPACECRAFT_FIELDS + _INSTRUMENT_FIELDS

# Spacecraft frames the Libera FK defines. Libera flies on JPSS-4; NOAA-20 is the alternate
# bus configuration carried in the same kernel set.
SPACECRAFT_OBSERVERS = ("JPSS4_SC", "NOAA20_SC")
DEFAULT_SPACECRAFT_OBSERVER = "JPSS4_SC"

# The four radiometer channel frames the Libera FK defines. All four are currently identity
# rotations relative to ``LIBERA_EL_COORD`` -- the channels are co-boresighted, so geometry
# computed for any one of them is valid for all four, and one query suffices. There is no
# generic ``LIBERA_RAD`` frame to name instead. Should the FK ever carry real per-channel
# boresight offsets, each channel would need its own query and the product would need
# per-channel geometry fields.
INSTRUMENT_OBSERVERS = ("LIBERA_SW_RAD", "LIBERA_LW_RAD", "LIBERA_TOT_RAD", "LIBERA_SSW_RAD")
DEFAULT_INSTRUMENT_OBSERVER = "LIBERA_SW_RAD"


def _validate_observer(observer: str, allowed: tuple[str, ...], role: str) -> None:
    """Reject an observer frame that is not a known Libera frame for this role.

    A typo'd frame already fails inside SPICE, but a *valid* frame used in the wrong role
    (the WFOV camera as the instrument, say) would silently produce correct-looking geometry
    for the wrong optic. This turns that into an error at the call site.

    Parameters
    ----------
    observer : str
        Requested SPICE frame name.
    allowed : tuple[str, ...]
        Frame names valid for this role.
    role : str
        Human-readable role name, used in the error message.

    Raises
    ------
    ValueError
        If ``observer`` is not in ``allowed``.
    """
    if observer not in allowed:
        raise ValueError(
            f"Unsupported {role} observer {observer!r}; expected one of {', '.join(allowed)}. "
            "Geometry for other frames is not supported by the L1B product definition."
        )


def _spice_error_message(err: SpiceyError) -> str:
    """User-facing description of a SPICE failure.

    Delegates to curryer's SPICE-error classifier, which maps the NAIF short name to a
    plain-language cause. Short names it does not recognize degrade to a generic
    ``"SPICE reported an error."`` summary rather than raising, and the short name and
    failing routine are appended either way -- so every ``SpiceyError`` yields a usable
    message and no additional exception handling is needed here.
    """
    return spice_error_message(err)


def calculate_lat_lon_altitude(
    kernel_manager: KernelManager,
    time_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate instrument geolocation (latitude/longitude/altitude).

    Instrument lat/lon/alt come from the :data:`DEFAULT_INSTRUMENT_OBSERVER` boresight
    ellipsoid intersection; the radiometer channels are co-boresighted, so this is the
    ground point for all four. The subsatellite point and the other geometry fields come
    from :func:`calculate_geometry`.

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
        u_gps_times,
        sp.obj.Body(DEFAULT_INSTRUMENT_OBSERVER, frame=True),
        give_geodetic_output=True,
        give_lat_lon_in_degrees=True,
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


def _query_geometry(
    observer: str,
    u_gps_times: np.ndarray,
    fields: tuple,
    require_coverage: bool,
    coverage_fields: tuple = (),
    **data_kwargs,
) -> pd.DataFrame:
    """
    One curryer ``GeometryData`` query, surfacing SPICE failures as readable errors.

    Extra keyword arguments (e.g. ``attitude_frame``) are forwarded to ``GeometryData``.

    Parameters
    ----------
    observer : str
        SPICE body name for the observer.
    u_gps_times : np.ndarray
        Query times in uGPS.
    fields : tuple of GeometryField
        Fields to compute for this observer.
    require_coverage : bool
        If True, raise when the coverage fields are entirely NaN -- the kernels do not cover
        the granule at all (a misconfiguration). The instrument observer passes False, since
        its fields are legitimately all-NaN in ``jpss_only`` mode.
    coverage_fields : tuple of GeometryField, optional
        Fields whose all-NaN state signals no coverage; defaults to every requested field. The
        spacecraft observer restricts this to an ephemeris-derived field, since the subsolar
        point is computed from the Sun ephemeris alone and stays finite even when the
        spacecraft kernels miss the granule.

    Returns
    -------
    pd.DataFrame
        The requested fields' columns, indexed by uGPS.

    Raises
    ------
    RuntimeError
        If the curryer SPICE query fails outright (e.g. an unparsable time or a missing
        kernel), or -- when ``require_coverage`` -- if it returns no coverage at all. Both
        carry a parsed, user-facing description of the cause.
    """
    try:
        result = geometry.GeometryData(observer, **data_kwargs).get_geometry(u_gps_times, fields=list(fields))
    except SpiceyError as err:
        raise RuntimeError(f"curryer geometry query failed for {observer!r}: {_spice_error_message(err)}") from err
    if require_coverage:
        coverage_columns = [column for field in (coverage_fields or fields) for column in field.columns]
        if bool(result[coverage_columns].isna().to_numpy().all()):
            raise RuntimeError(
                f"curryer geometry returned no coverage for observer {observer!r} over the granule; "
                "check that the SPICE kernels cover the requested times."
            )
    return result


def calculate_geometry(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    spacecraft_observer: str = DEFAULT_SPACECRAFT_OBSERVER,
    instrument_observer: str = DEFAULT_INSTRUMENT_OBSERVER,
) -> pd.DataFrame:
    """
    Compute the geometry fields via curryer ``GeometryData``.

    curryer's selective-compute registry queries each SPICE input once, vectorized, with
    coverage gaps surfaced as NaN. Two calls, joined on the shared uGPS index:

    - the **spacecraft** observer yields the fields needing only its ephemeris and body
      attitude (subsatellite and subsolar points, satellite radius, altitude, inertial
      (J2000) position and velocity, and the Earth-fixed attitude quaternion);
    - the **instrument** observer yields the boresight geometry (viewing and solar zenith /
      azimuth, relative azimuth, cone and clock angles and their rates, along/cross-track
      look angles, and the inertial boresight vector).

    The split is by necessity, not preference. The spacecraft observer resolves a valid
    position in every mode, whereas the boresight fields require the instrument frame. In
    ``jpss_only`` (no motor CK) the instrument frame does not resolve -- not even its
    position -- so its fields are NaN, but the spacecraft fields still come through, which
    is what the ``jpss_only`` geolocation (subsatellite point) depends on. With full
    pointing the two observers give identical spacecraft-level values (the instrument
    sits at the spacecraft's SPK position).

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the spacecraft SPK, instrument FK/IK, and (for the boresight
        fields) the pointing CK furnished.
    timestamps : np.ndarray
        Radiometer timestamps on the L1B output time grid.
    spacecraft_observer : str
        SPICE frame for the spacecraft, one of :data:`SPACECRAFT_OBSERVERS`. Default
        ``"JPSS4_SC"``.
    instrument_observer : str
        SPICE frame for the instrument, one of :data:`INSTRUMENT_OBSERVERS`. Default
        ``"LIBERA_SW_RAD"``. The radiometer channels are co-boresighted, so this choice does
        not currently change the result -- see :data:`INSTRUMENT_OBSERVERS`.

    Returns
    -------
    pd.DataFrame
        Indexed by uGPS; the curryer field columns for both observers, joined.

    Raises
    ------
    ValueError
        If either observer is not a known Libera frame for its role.
    RuntimeError
        If a curryer SPICE query fails outright (e.g. an unparsable time or a missing
        kernel), or if the spacecraft observer returns no coverage at all. Both carry a
        parsed, user-facing description of the cause.
    """
    _validate_observer(spacecraft_observer, SPACECRAFT_OBSERVERS, "spacecraft")
    _validate_observer(instrument_observer, INSTRUMENT_OBSERVERS, "instrument")
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    # Spacecraft fields resolve in every mode, so all-NaN means the kernels miss the granule. The
    # attitude quaternion is Earth-fixed (product convention); the inertial fields keep J2000.
    spacecraft = _query_geometry(
        spacecraft_observer,
        u_gps_times,
        _SPACECRAFT_FIELDS,
        require_coverage=True,
        coverage_fields=(geometry.GeometryField.SUBSATELLITE,),
        attitude_frame=spatial.EARTH_FRAME,
    )
    # Instrument fields are legitimately all-NaN in jpss_only (no motor CK).
    instrument = _query_geometry(instrument_observer, u_gps_times, _INSTRUMENT_FIELDS, require_coverage=False)
    return spacecraft.join(instrument)


def calculate_start_of_hour_state(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    observer: str = "JPSS4_SC",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spacecraft inertial (J2000) position and velocity at each top-of-hour of the granule's UTC day.

    The product carries the spacecraft state on a fixed 24-hour grid (``N_HOURS``), not on
    the radiometer time grid. "Start of hour" is a mission timetag convention, so the
    hour-boundary epochs are built here and curryer is queried at exactly those times --
    curryer's ``GeometryData`` has no notion of an hour boundary.

    The 24 epochs are the top of each hour (00:00 .. 23:00 UTC) of the day the granule
    *starts* in; a granule spanning midnight still reports its start day. Hours outside
    SPK coverage come back NaN, per the curryer fill contract.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the spacecraft SPK furnished.
    timestamps : np.ndarray
        Radiometer timestamps; only the first is used, to pick the UTC day.
    observer : str
        SPICE body name for the spacecraft. Default ``"JPSS4_SC"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(position, velocity)`, each shape `(24, 3)` in the J2000 inertial frame, km and km/s.
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    day_start = pd.Timestamp(np.asarray(timestamps, dtype="datetime64[ns]")[0]).normalize()
    hour_epochs = pd.date_range(day_start, periods=24, freq="h")
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(hour_epochs), "iso")

    fields = (geometry.GeometryField.SC_POSITION_INERTIAL, geometry.GeometryField.SC_VELOCITY_INERTIAL)
    hourly = geometry.GeometryData(observer).get_geometry(u_gps_times, fields=list(fields))
    position = hourly[list(geometry.GeometryField.SC_POSITION_INERTIAL.columns)].to_numpy()
    velocity = hourly[list(geometry.GeometryField.SC_VELOCITY_INERTIAL.columns)].to_numpy()
    logger.debug("Start-of-hour state: %d hourly epochs from %s", len(hour_epochs), day_start)
    return position, velocity


def subsatellite_lat_lon_alt(geometry_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract subsatellite ``lat``/``lon``/``alt`` from the geometry fields.

    The subsatellite ground point's latitude/longitude plus the spacecraft geodetic
    altitude, in the columns the product packager consumes. Used as the geolocation in
    ``jpss_only`` mode, where no instrument pointing is available.

    Parameters
    ----------
    geometry_data : pd.DataFrame
        Output of :func:`calculate_geometry`.

    Returns
    -------
    pd.DataFrame
        Columns ``lat``, ``lon``, ``alt``, on the input's uGPS index.
    """
    subsatellite_lat, subsatellite_lon, _ = geometry.GeometryField.SUBSATELLITE.columns
    (spacecraft_altitude,) = geometry.GeometryField.SC_ALTITUDE.columns
    return pd.DataFrame(
        {
            "lat": geometry_data[subsatellite_lat].to_numpy(),
            "lon": geometry_data[subsatellite_lon].to_numpy(),
            "alt": geometry_data[spacecraft_altitude].to_numpy(),
        },
        index=geometry_data.index,
    )


def create_placeholder_geometry(n_samples: int) -> pd.DataFrame:
    """
    Placeholder geometry fields for ``use_geo`` false mode.

    Mirrors :func:`calculate_geometry`'s columns filled with the product fill values, so the
    packager always reads a geometry DataFrame and never branches on ``None``. Angular fields
    (subsatellite / subsolar points) use -999; distance fields (radius, altitude) use -9999.

    Parameters
    ----------
    n_samples : int
        Number of samples on the L1B output time grid.

    Returns
    -------
    pd.DataFrame
        One column per :data:`_GEOMETRY_FIELDS` output column, filled with the fill value.
    """
    angle_fill = np.full(n_samples, -999.0, dtype=np.float32)
    distance_fill = np.full(n_samples, -9999.0, dtype=np.float64)
    distance_columns = {
        geometry.GeometryField.SC_RADIUS.columns[0],
        geometry.GeometryField.SC_ALTITUDE.columns[0],
        *geometry.GeometryField.SC_POSITION_INERTIAL.columns,
    }
    data = {
        column: (distance_fill if column in distance_columns else angle_fill)
        for field in _GEOMETRY_FIELDS
        for column in field.columns
    }
    return pd.DataFrame(data)


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
