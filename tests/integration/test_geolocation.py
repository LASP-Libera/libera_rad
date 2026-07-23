"""Integration tests for geolocation using static and geolocation kernels."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad.geolocation import calculate_geometry, calculate_lat_lon_altitude


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
