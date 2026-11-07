"""Integration tests for geolocation using static and geolocation kernels."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from curryer import spicierpy as sp
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad.constants import DEFAULT_INSTRUMENT_OBSERVER, MOON_FOV_BUFFER_DEG, MOON_IN_FOV_FILL_VALUE
from libera_rad.geolocation import (
    add_moon_boresight_offsets,
    calculate_geometry,
    calculate_lat_lon_altitude,
    moon_in_field_of_view,
)


def _dynamic_kernel_paths(kernel_dir: Path) -> list[Path]:
    return sorted(f for f in kernel_dir.iterdir() if f.is_file())


def test_geolocation_from_kernels(test_dynamic_kernels_path):
    """
    Test geolocation computation using static and geolocation kernels from the Libera SDC tier 1 geolocation test.
    """
    start_time = datetime(2028, 1, 2, 0, 13, 47)
    end_time = datetime(2028, 1, 2, 0, 29, 12)

    km = KernelManager()
    km.load_libera_dynamic_kernels(
        _dynamic_kernel_paths(test_dynamic_kernels_path), needs_naif_kernels=True, needs_static_kernels=True
    )

    print(f"Calculating lat/lon/alt from {start_time} to {end_time}")
    time_range = pd.date_range(start_time, end_time, freq="10ms", inclusive="left")

    lat_lon_alt = calculate_lat_lon_altitude(km, time_range)

    print("Mean Lat: ", np.nanmean(lat_lon_alt["lat"]))
    print("Mean Lon: ", np.nanmean(lat_lon_alt["lon"]))

    clon, clat = 23.39, 28.55
    hist, xedge, yedge = np.histogram2d(
        lat_lon_alt["lon"],
        lat_lon_alt["lat"],
        bins=(13, 13),
        range=[[clon - 3, clon + 3], [clat - 3, clat + 3]],
    )
    idx = np.where(hist == hist.max())
    ix, iy = idx[0][0], idx[1][0]

    print(f"Expected focus point:    lon=[{clon}],          lat=[{clat}]")
    print(
        f"2D histogram max between lon=[{xedge[ix]:.3f}, {xedge[ix + 1]:.3f}],"
        f" lat=[{yedge[iy]:.3f}, {yedge[iy + 1]:.3f}]"
    )
    assert ix == hist.shape[0] // 2, ix
    assert iy == hist.shape[0] // 2, iy


def test_surface_geometry_from_kernels(test_dynamic_kernels_path):
    """Boresight surface geometry from real kernels must be physically self-consistent.

    Exercises :func:`calculate_geometry` end to end against SPICE rather than mocks, and
    checks the invariants that would catch a frame, unit, or convention error: declared
    ranges, the zenith/cone-angle relationship, and the relative-azimuth identity.
    """
    km = KernelManager()
    km.load_libera_dynamic_kernels(
        _dynamic_kernel_paths(test_dynamic_kernels_path), needs_naif_kernels=True, needs_static_kernels=True
    )

    timestamps = pd.date_range(datetime(2028, 1, 2, 0, 13, 47), periods=500, freq="100ms").to_numpy()
    geometry_data = calculate_geometry(km, timestamps)

    finite = np.isfinite(geometry_data["viewing_zenith"].to_numpy())
    assert finite.any(), "No finite surface geometry computed from the test kernels"

    # Bounds on curryer's raw values, which are wider than the L1B product's declared
    # valid_range -- the cone angle exceeds 90 degrees for limb-ward pointing. Conformance to
    # the narrower product ranges is asserted against the written product in test_l1b.py.
    for column, (low, high) in {
        "viewing_zenith": (0.0, 180.0),
        "solar_zenith": (0.0, 180.0),
        "viewing_azimuth": (0.0, 360.0),
        "solar_azimuth": (0.0, 360.0),
        "cone_angle": (0.0, 180.0),
    }.items():
        vals = geometry_data[column].to_numpy()
        vals = vals[np.isfinite(vals)]
        assert len(vals) > 0, f"{column} produced no finite values"
        assert np.all((vals >= low) & (vals <= high)), (
            f"{column} outside [{low}, {high}]: min={vals.min()}, max={vals.max()}"
        )

    # The surface viewing zenith is measured at the footprint and the cone angle at the
    # spacecraft, so for a spherical Earth sin(zenith) = (r_sat / r_surface) * sin(cone), making
    # the zenith the larger of the two. A frame or sign inversion breaks this badly, which is
    # what we are guarding against.
    #
    # The two are not referenced to the same vertical, though: the cone angle is measured from
    # the *geocentric* nadir (-sc_position) while the zenith is *geodetic*, and on an oblate
    # Earth those differ by up to ~0.19 degrees. Away from nadir the ~1.13x amplification
    # dominates that offset comfortably; within a few degrees of nadir it does not, and the
    # ordering can legitimately invert. Restrict the check to the off-nadir regime where the
    # inequality is actually implied by the geometry -- that is also where an inversion would
    # be large enough to matter.
    viewing_zenith = geometry_data["viewing_zenith"].to_numpy()
    cone_angle = geometry_data["cone_angle"].to_numpy()
    off_nadir = np.isfinite(viewing_zenith) & np.isfinite(cone_angle) & (cone_angle > 5.0)
    assert off_nadir.any(), "No off-nadir samples available to check the zenith/cone-angle relation"
    assert np.all(viewing_zenith[off_nadir] >= cone_angle[off_nadir]), (
        "Off-nadir surface viewing zenith must exceed the spacecraft cone angle; "
        f"worst case zenith={viewing_zenith[off_nadir].min()}, cone={cone_angle[off_nadir].max()}"
    )

    # curryer uses the CERES BDS R3V4 origin: mod(viewing_azimuth - solar_azimuth + 180, 360),
    # which puts the Sun at 180 rather than 0. This is the unfolded, lossless value -- the
    # CERES [0, 180] fold is a separate lossy step we do not apply.
    expected = np.mod(
        geometry_data["viewing_azimuth"].to_numpy() - geometry_data["solar_azimuth"].to_numpy() + 180.0,
        360.0,
    )
    actual = geometry_data["relative_azimuth"].to_numpy()
    valid = np.isfinite(expected) & np.isfinite(actual)
    # Compare on the circle so the 0/360 wrap does not register as a 360-degree error.
    separation = np.abs(np.mod(actual[valid] - expected[valid] + 180.0, 360.0) - 180.0)
    assert np.all(separation < 1e-3), f"relative_azimuth inconsistent with its components, max diff {separation.max()}"


def test_dynamic_kernels_materialize_into_cache(monkeypatch, tmp_path, test_dynamic_kernels_path):
    """Dynamic kernels should be materialized into the user cache via KernelFileCache."""
    monkeypatch.setattr(spice_utils.caching, "get_local_cache_dir", lambda: tmp_path)

    kernel_files = sorted(
        [p for p in test_dynamic_kernels_path.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"}],
        key=lambda p: p.name,
    )
    assert kernel_files, f"No dynamic kernels found under {test_dynamic_kernels_path}"
    sources = kernel_files[:2]

    km = KernelManager(cache_timeout_days=7)
    km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)

    for src in sources:
        assert (tmp_path / src.name).is_file(), f"Expected cached kernel missing: {src.name}"


def test_lunar_geometry_from_kernels(test_dynamic_kernels_path):
    """Lunar boresight geometry from real kernels must be physically self-consistent.

    Exercises the curryer lunar fields, :func:`add_moon_boresight_offsets` and
    :func:`moon_in_field_of_view` end to end against SPICE rather than mocks, checking the
    invariants that would catch a frame, unit, or convention error.
    """
    km = KernelManager()
    km.load_libera_dynamic_kernels(
        _dynamic_kernel_paths(test_dynamic_kernels_path), needs_naif_kernels=True, needs_static_kernels=True
    )

    timestamps = pd.date_range(datetime(2028, 1, 2, 0, 13, 47), periods=500, freq="100ms").to_numpy()
    geometry_data = calculate_geometry(km, timestamps)
    geometry_data = add_moon_boresight_offsets(km, geometry_data)

    direction = geometry_data[["moon_direction_x", "moon_direction_y", "moon_direction_z"]].to_numpy()
    boresight_angle = geometry_data["moon_boresight_angle"].to_numpy()
    azimuth_offset = geometry_data["moon_azimuth_offset"].to_numpy()
    elevation_offset = geometry_data["moon_elevation_offset"].to_numpy()
    angular_radius = geometry_data["moon_angular_radius"].to_numpy()
    distance = geometry_data["moon_distance"].to_numpy()

    # Guard: a granule of all-NaN lunar geometry would pass every check below vacuously.
    assert np.isfinite(boresight_angle).any()

    # A spacecraft position in the wrong units (the meters-not-km delivery bug) throws the
    # Moon direction off entirely, so anchor on the radius before trusting the angles.
    assert np.allclose(geometry_data["spacecraft_radius"].to_numpy(), 7000.0, atol=500.0)

    # Declared product ranges.
    assert np.nanmin(boresight_angle) >= 0.0
    assert np.nanmax(boresight_angle) <= 180.0
    assert np.nanmin(azimuth_offset) >= -180.0
    assert np.nanmax(azimuth_offset) <= 180.0
    assert np.nanmin(elevation_offset) >= -90.0
    assert np.nanmax(elevation_offset) <= 90.0

    # Physical lunar range and apparent size. Bounds are topocentric, not the familiar
    # geocentric figures: the observer is the spacecraft, which sits up to an Earth radius
    # plus altitude nearer or farther than Earth's center, widening perigee/apogee to
    # roughly [349e3, 413e3] km and the disk radius to [0.241, 0.285] deg.
    assert np.nanmin(distance) > 3.45e5
    assert np.nanmax(distance) < 4.15e5
    assert np.nanmin(angular_radius) > 0.240
    assert np.nanmax(angular_radius) < 0.286

    # The offset pair and the total angle describe one direction.
    np.testing.assert_allclose(
        np.cos(np.radians(boresight_angle)),
        np.cos(np.radians(azimuth_offset)) * np.cos(np.radians(elevation_offset)),
        atol=1e-9,
    )

    # curryer reports the direction in the IK's FOV frame, which is what lets the offsets be
    # taken against the boresight with no rotation: the separation must come back out as a
    # plain dot product. A direction reported in some other frame would fail here.
    finite = np.isfinite(direction).all(axis=-1)
    np.testing.assert_allclose(np.linalg.norm(direction[finite], axis=-1), 1.0, rtol=1e-12)
    boresight = sp.ext.instrument_fov(DEFAULT_INSTRUMENT_OBSERVER).boresight
    expected_angle = np.degrees(np.arccos(direction[finite] @ (boresight / np.linalg.norm(boresight))))
    np.testing.assert_allclose(boresight_angle[finite], expected_angle, atol=1e-9)

    # The flag agrees with the threshold it documents, and is the fill value exactly where
    # the lunar geometry is missing rather than defaulting to "not in view".
    flags = moon_in_field_of_view(km, geometry_data)
    fov_half_angle = sp.ext.instrument_fov(DEFAULT_INSTRUMENT_OBSERVER).half_angle(degrees=True)
    expected_in_view = boresight_angle <= fov_half_angle + angular_radius + float(MOON_FOV_BUFFER_DEG)
    missing = np.isnan(boresight_angle)
    np.testing.assert_array_equal(flags[missing], MOON_IN_FOV_FILL_VALUE)
    np.testing.assert_array_equal(flags[~missing], expected_in_view[~missing].astype(np.int8))
