"""Helpers for the L1B observation geometry notebook: kernels, vector math, 3D drawing and checks."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from curryer import spicierpy as sp
from curryer.compute import spatial
from curryer.compute.constants import KM_PER_ASTRONOMICAL_UNIT, WGS84_SEMI_MAJOR_AXIS_KM, WGS84_SEMI_MINOR_AXIS_KM
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad.geolocation import calculate_scan_rates

A_EARTH, B_EARTH = WGS84_SEMI_MAJOR_AXIS_KM, WGS84_SEMI_MINOR_AXIS_KM
AU_KM = KM_PER_ASTRONOMICAL_UNIT
OMEGA_EARTH = np.array([0.0, 0.0, 7.292115e-5])  # WGS84 Earth rotation rate, rad/s

# Apparent lunar radius seen from low Earth orbit, degrees; bounds the in-view threshold check.
MOON_RADIUS_RANGE_DEG = (0.24, 0.29)


# --- Kernels ------------------------------------------------------------------------------


def load_kernels(kernel_dir: Path) -> KernelManager:
    """Furnish the dynamic kernels in `kernel_dir` plus the NAIF and static Libera kernels.

    Kernels already in the libera_utils cache warn about a basename collision on every load. The
    cached file is a copy of the same kernel, so the warning and its log message are silenced.
    """
    sources = [str(path) for path in sorted(kernel_dir.iterdir()) if path.suffix in (".bc", ".bsp")]
    kernel_log = logging.getLogger("libera_utils.libera_spice.kernel_manager")
    previous_level = kernel_log.level
    kernel_manager = KernelManager()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*basename conflict.*")
        kernel_log.setLevel(logging.ERROR)
        try:
            kernel_manager.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        finally:
            kernel_log.setLevel(previous_level)
    kernel_manager.ensure_known_kernels_are_furnished()
    return kernel_manager


# --- Vector geometry ----------------------------------------------------------------------


def unit(v):
    """Normalise along the last axis."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_between(a, b):
    """Angle in degrees between two directions, accurate when they nearly coincide."""
    a, b = unit(np.asarray(a, dtype="float64")), unit(np.asarray(b, dtype="float64"))
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))


def wrapped_difference(a, b):
    """Absolute difference between two angle arrays in degrees, wrapped to [0, 180]."""
    diff = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def geodetic_to_ecef(lat_deg, lon_deg):
    """Earth-fixed position in km of a point on the WGS84 ellipsoid."""
    lat_deg, lon_deg = np.broadcast_arrays(np.asarray(lat_deg, "float64"), np.asarray(lon_deg, "float64"))
    lon_lat_alt = np.stack([lon_deg, lat_deg, np.zeros_like(lat_deg)], axis=-1)
    return np.asarray(spatial.geodetic_to_ecef(lon_lat_alt.reshape(-1, 3))).reshape(lon_lat_alt.shape)


def geodetic_normal(lat_deg, lon_deg):
    """Outward ellipsoid normal, the local vertical, at a geodetic latitude and longitude."""
    phi, lam = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def east_north_up(lat_deg, lon_deg):
    """Local east, north and up unit vectors at geodetic points."""
    up = geodetic_normal(lat_deg, lon_deg)
    east = unit(np.cross(np.broadcast_to([0.0, 0.0, 1.0], up.shape), up))
    return east, np.cross(up, east), up


def orbital_frame(positions, forwards):
    """Per-sample nadir, in-track and cross-track unit vectors, in the frame the inputs are in."""
    nadir = -unit(positions)
    in_track = unit(forwards - (forwards * nadir).sum(-1)[..., None] * nadir)
    return nadir, in_track, np.cross(nadir, in_track)


def attitude_matrices(quaternions):
    """Body-to-Earth-fixed rotation matrices from scalar-first SPICE quaternions; columns are body axes."""
    return np.stack([sp.q2m(q) for q in unit(np.asarray(quaternions, dtype="float64"))])


def turning_points(angle):
    """Sample indices where a scan angle reverses direction, skipping NaN."""
    valid = np.flatnonzero(np.isfinite(angle))
    return valid[np.flatnonzero(np.diff(np.sign(np.diff(angle[valid]))) != 0) + 1]


def finite_max(a, b):
    """Largest absolute difference where both inputs are finite, and how many values that covers."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64"))
    both = np.isfinite(a) & np.isfinite(b)
    return (float(np.max(np.abs(a[both] - b[both]))) if both.any() else np.nan), int(both.sum())


def finite_wrapped_max(a, b):
    """`finite_max` for angles in degrees, taking the shorter way round the circle."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64"))
    both = np.isfinite(a) & np.isfinite(b)
    return (float(wrapped_difference(a[both], b[both]).max()) if both.any() else np.nan), int(both.sum())


# --- 3D drawing ---------------------------------------------------------------------------

HALO = {"bbox": dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)}


def axes3d(fig, *subplot_spec):
    """3D axes with matplotlib's depth sorting off, so explicit zorder decides what covers what."""
    return fig.add_subplot(*subplot_spec, projection="3d", computed_zorder=False)


def frame3d(ax, camera, half_width, center=(0.0, 0.0, 0.0), zoom=1.0):
    """Look along `camera`, give the three axes one scale, and hide the box."""
    d = unit(np.asarray(camera, dtype="float64"))
    ax.view_init(elev=float(np.degrees(np.arcsin(d[2]))), azim=float(np.degrees(np.arctan2(d[1], d[0]))))
    for set_limit, middle in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=True):
        set_limit(middle - half_width, middle + half_width)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()


def arc_points(from_vec, to_vec, radius, origin=(0.0, 0.0, 0.0), n=60):
    """Points along the shorter great-circle arc between two directions, about `origin`."""
    a, b = unit(np.asarray(from_vec, dtype="float64")), unit(np.asarray(to_vec, dtype="float64"))
    perpendicular = b - (b @ a) * a
    length = np.linalg.norm(perpendicular)
    if length < 1e-12:
        return np.full((n, 3), np.nan)
    sweep = np.linspace(0.0, np.arccos(np.clip(a @ b, -1.0, 1.0)), n)
    return np.asarray(origin, dtype="float64") + radius * (
        np.cos(sweep)[:, None] * a + np.sin(sweep)[:, None] * (perpendicular / length)
    )


def arc3d(ax, origin, from_vec, to_vec, radius, **kwargs):
    """Draw the shorter great-circle arc between two directions about `origin`."""
    points = arc_points(from_vec, to_vec, radius, origin)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def label3d(ax, point, text, **kwargs):
    """Text at a 3D point, centred by default."""
    kwargs.setdefault("ha", "center")
    ax.text(*np.asarray(point, dtype="float64"), text, **kwargs)


# --- Consistency checks -------------------------------------------------------------------


def check_geometry_fields(
    ds: xr.Dataset, satellite_ecef: np.ndarray, fov_half_angle_deg: float, moon_buffer_deg: float
) -> pd.DataFrame:
    """Check the geometry fields of one full-pointing L1B granule against each other.

    Parameters
    ----------
    ds : xr.Dataset
        The granule, opened with ``mask_and_scale`` so fill values are NaN.
    satellite_ecef : np.ndarray
        ``Satellite_Position`` rotated into ITRF93, shape ``(N, 3)``, km.
    fov_half_angle_deg : float
        Radiometer field-of-view half angle from the IK, degrees.
    moon_buffer_deg : float
        ``MOON_FOV_BUFFER_DEG``, the buffer the lunar in-view flag adds.

    Returns
    -------
    pd.DataFrame
        One row per check: residual, tolerance, unit, samples compared and a pass/REVIEW verdict.
        A ``jpss_only`` granule fails the surface-mask check by design, since its Latitude and
        Longitude hold the subsatellite point while the boresight surface angles are NaN.
    """
    rows = []

    def record(check, result, tolerance, unit_name, note=""):
        """Append one check; `result` is `(residual, samples compared)`."""
        residual, n = result
        verdict = "pass" if np.isfinite(residual) and residual <= tolerance else "REVIEW"
        rows.append(
            dict(check=check, residual=residual, tolerance=tolerance, unit=unit_name, n=n, verdict=verdict, note=note)
        )

    def field(name):
        return ds[name].values.astype("float64")

    time = ds["radiometer_time"].values
    seconds = (time - time[0]) / np.timedelta64(1, "s")
    lat, lon = field("Latitude"), field("Longitude")
    sub_lat, sub_lon = field("Subsatellite_Latitude"), field("Subsatellite_Longitude")
    cone, cross_track = field("Cone_Angle"), field("Cross_Track_Angle")
    position = field("Satellite_Position")
    on_earth = np.isfinite(lat)

    # Definitions.
    for lat_name, colat_name in (
        ("Latitude", "Colatitude"),
        ("Subsatellite_Latitude", "Subsatellite_Colatitude"),
        ("Subsolar_Latitude", "Subsolar_Colatitude"),
    ):
        record(f"{colat_name} == 90 - {lat_name}", finite_max(90.0 - field(lat_name), field(colat_name)), 1e-4, "deg")
    norm = np.linalg.norm(field("Line_Of_Sight"), axis=1)
    record("|Line_Of_Sight| == 1", finite_max(norm, np.ones_like(norm)), 1e-6, "-")
    quaternion_norm = np.linalg.norm(np.stack([field(f"Satellite_Attitude_Q{i}") for i in range(4)], axis=1), axis=1)
    record("|Satellite_Attitude| == 1", finite_max(quaternion_norm, np.ones_like(quaternion_norm)), 1e-6, "-")
    record(
        "Radius == |Satellite_Position|",
        finite_max(field("Radius_of_Satellite_from_Center_of_Earth"), np.linalg.norm(position, axis=1)),
        1e-6,
        "km",
    )
    predicted_raa = np.mod(
        field("Viewing_Azimuth_Surface_WRT_North") - field("Solar_Azimuth_Surface_WRT_North") + 180.0, 360.0
    )
    record(
        "Relative_Azimuth == mod(viewing - solar + 180, 360)",
        finite_wrapped_max(predicted_raa, field("Relative_Azimuth_Surface")),
        1e-3,
        "deg",
    )

    # Pointing: the look angles, the J2000 position and the Earth-fixed surface point must agree.
    along, cross = np.radians(field("Along_Track_Angle")), np.radians(cross_track)
    record(
        "Cone_Angle == atan(hypot(tan(along), tan(cross)))",
        finite_max(np.degrees(np.arctan(np.hypot(np.tan(along), np.tan(cross)))), cone),
        1e-4,
        "deg",
    )
    satellite_geodetic = spatial.ecef_to_geodetic(satellite_ecef)
    lat_check, lon_check = (
        finite_max(satellite_geodetic[:, 1], sub_lat),
        finite_wrapped_max(satellite_geodetic[:, 0], sub_lon),
    )
    record(
        "rotated Satellite_Position lands on the subsatellite point",
        (max(lat_check[0], lon_check[0]), lat_check[1]),
        1e-4,
        "deg",
        "ties the J2000 position to the Earth-fixed fields",
    )
    surface = geodetic_to_ecef(lat, lon)
    record(
        "Cone_Angle == angle(surface - spacecraft, -spacecraft)",
        finite_max(angle_between(surface - satellite_ecef, -satellite_ecef), cone),
        2e-4,
        "deg",
        "ties Latitude/Longitude to the boresight",
    )

    # Surface angles, rebuilt in each sample's local horizon frame.
    east, north, up = east_north_up(lat, lon)
    to_spacecraft = unit(satellite_ecef - surface)
    record(
        "Viewing_Zenith == angle(to spacecraft, local vertical)",
        finite_max(angle_between(to_spacecraft, up), field("Viewing_Zenith_Surface")),
        1e-3,
        "deg",
    )
    vaa = np.degrees(np.arctan2((to_spacecraft * east).sum(1), (to_spacecraft * north).sum(1))) % 360.0
    steep = field("Viewing_Zenith_Surface") > 1.0
    record(
        "Viewing_Azimuth == azimuth of the spacecraft in the horizon",
        finite_wrapped_max(np.where(steep, vaa, np.nan), field("Viewing_Azimuth_Surface_WRT_North")),
        2e-2,
        "deg",
        "viewing zenith > 1 deg; azimuth is ill-conditioned near zenith",
    )
    phi, lam = np.radians(lat), np.radians(lon)
    sun_phi, sun_lam = np.radians(field("Subsolar_Latitude")), np.radians(field("Subsolar_Longitude"))
    cos_sza = np.sin(phi) * np.sin(sun_phi) + np.cos(phi) * np.cos(sun_phi) * np.cos(lam - sun_lam)
    record(
        "Solar_Zenith == great-circle angle to the subsolar point",
        finite_max(np.degrees(np.arccos(np.clip(cos_sza, -1, 1))), field("Solar_Zenith_Surface")),
        5e-3,
        "deg",
        "residual is solar parallax, R_earth / 1 AU",
    )

    # Earth limb: surface fields are NaN where pointing is missing or the boresight is past the limb.
    surface_fields = [
        "Latitude",
        "Longitude",
        "Colatitude",
        "Altitude",
        "Viewing_Zenith_Surface",
        "Solar_Zenith_Surface",
        "Viewing_Azimuth_Surface_WRT_North",
        "Solar_Azimuth_Surface_WRT_North",
        "Relative_Azimuth_Surface",
    ]
    mismatched = sum(not np.array_equal(np.isnan(field(name)), ~on_earth) for name in surface_fields)
    record("surface fields share one NaN mask", (float(mismatched), len(surface_fields)), 0, "fields")
    geocentric_radius = np.linalg.norm(geodetic_to_ecef(sub_lat, sub_lon), axis=1)
    limb = np.degrees(np.arcsin(geocentric_radius / np.linalg.norm(position, axis=1)))
    pointed = np.isfinite(cone)
    misplaced = pointed & ((on_earth & (cone > limb + 0.2)) | (~on_earth & (cone < limb - 0.2)))
    record(
        "on or off Earth agrees with the limb to 0.2 deg",
        (float(misplaced.sum()), int(pointed.sum())),
        0,
        "samples",
        f"{int((~pointed).sum())} samples have no pointing at all",
    )

    # Motor encoders against the derived pointing.
    elevation, azimuth = field("Elevation"), field("Azimuth")
    paired = np.isfinite(elevation) & np.isfinite(cross_track)
    offset = float(np.median(cross_track[paired] + elevation[paired]))
    record(
        "Cross_Track_Angle == -Elevation + constant",
        finite_max(cross_track + elevation, np.full_like(cross_track, offset)),
        5e-3,
        "deg",
        f"offset {offset:+.4f} deg: FK boresight tilt plus geodetic-nadir attitude",
    )

    # Scan rates.
    cone_rate, clock_rate = field("Cone_Angle_Rate"), field("Clock_Angle_Rate")
    rederived = [np.where(rate == -999.0, np.nan, rate) for rate in calculate_scan_rates(azimuth, elevation, time)]
    record(
        "Cone_Angle_Rate == calculate_scan_rates(Azimuth, Elevation)",
        finite_max(rederived[0].astype("float64"), cone_rate),
        5e-3,
        "deg/s",
        "float32 encoder angles",
    )
    record(
        "Clock_Angle_Rate == calculate_scan_rates(Azimuth, Elevation)",
        finite_max(rederived[1].astype("float64"), clock_rate),
        5e-3,
        "deg/s",
        "float32 encoder angles; weak while the azimuth is parked",
    )
    off_nadir = np.isfinite(cone_rate[1:]) & (np.minimum(cone[:-1], cone[1:]) > 1.0)
    wrong_sign = np.count_nonzero(np.sign(cone_rate[1:][off_nadir]) != np.sign(np.diff(cone)[off_nadir]))
    record(
        "sign(Cone_Angle_Rate) == sign(d(Cone_Angle)) off nadir",
        (float(wrong_sign), int(off_nadir.sum())),
        0,
        "samples",
        "catches a gross nadir or sign error",
    )
    encoder_known = np.isfinite(azimuth) & np.isfinite(elevation)
    on_cadence = np.abs(np.diff(seconds) - 0.01) <= 0.005
    cannot_form = np.ones(len(time), dtype=bool)
    cannot_form[1:] = ~(encoder_known[:-1] & encoder_known[1:] & on_cadence)
    misfilled = sum(np.count_nonzero(np.isnan(rate) != cannot_form) for rate in (cone_rate, clock_rate))
    record(
        "scan rates filled exactly where they cannot be formed",
        (float(misfilled), len(time)),
        0,
        "samples",
        "missing encoder angle or off-cadence interval",
    )
    record(
        "Satellite_Velocity == d(Satellite_Position)/dt",
        finite_max(np.gradient(position, seconds, axis=0)[1:-1], field("Satellite_Velocity")[1:-1]),
        1e-3,
        "km/s",
        "finite difference",
    )

    # Lunar fields.
    moon_angle = field("Moon_Boresight_Angle")
    moon_az, moon_el = np.radians(field("Moon_Azimuth_Offset")), np.radians(field("Moon_Elevation_Offset"))
    record(
        "cos(Moon_Boresight_Angle) == cos(az offset) cos(el offset)",
        finite_max(np.degrees(np.arccos(np.clip(np.cos(moon_az) * np.cos(moon_el), -1, 1))), moon_angle),
        1e-3,
        "deg",
    )
    flag = field("Moon_In_Field_Of_View")
    low = fov_half_angle_deg + MOON_RADIUS_RANGE_DEG[0] + moon_buffer_deg
    high = fov_half_angle_deg + MOON_RADIUS_RANGE_DEG[1] + moon_buffer_deg
    wrong_flag = (
        ((flag == 1) & ~(moon_angle <= high))
        | ((flag == 0) & ~(moon_angle > low))
        | (np.isnan(flag) != np.isnan(moon_angle))
    )
    record(
        "Moon_In_Field_Of_View matches its threshold, fill where the Moon is NaN",
        (float(wrong_flag.sum()), len(flag)),
        0,
        "samples",
        f"threshold {low:.2f} to {high:.2f} deg",
    )

    # Declared ranges.
    outside = 0
    for name in ds.data_vars:
        declared = ds[name].attrs.get("valid_range")
        values = ds[name].values.astype("float64")
        if declared is not None and len(declared) == 2:
            outside += int(np.any((values < float(declared[0])) | (values > float(declared[1]))))
    record("every field inside its declared valid_range", (float(outside), len(ds.data_vars)), 0, "fields")

    return pd.DataFrame(rows)
