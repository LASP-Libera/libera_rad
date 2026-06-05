"""Unit tests for geolocation helpers."""

import numpy as np

from libera_rad import geolocation


def test_create_placeholder_azimuth_elevation():
    az, el = geolocation.create_placeholder_azimuth_elevation(4, fill_value=-999.0)
    assert az.shape == (4,)
    assert el.shape == (4,)
    assert np.all(az == np.float32(-999))
    assert np.all(el == np.float32(-999))


def test_create_jpss_only_motor_angles():
    az, el = geolocation.create_jpss_only_motor_angles(3)
    assert np.all(az == 0)
    assert np.all(el == 0)
    assert az.dtype == np.float32


def test_create_placeholder_surface_geometry_angles():
    angles = geolocation.create_placeholder_surface_geometry_angles(5, seed=42)
    assert angles["solar_zenith"].shape == (5,)
    assert np.all((angles["solar_zenith"] >= 10) & (angles["solar_zenith"] <= 80))
    assert np.all((angles["viewing_zenith"] >= 5) & (angles["viewing_zenith"] <= 70))
    assert np.all((angles["relative_azimuth"] >= 0) & (angles["relative_azimuth"] < 360))
    assert len(np.unique(angles["solar_zenith"])) > 1

    again = geolocation.create_placeholder_surface_geometry_angles(5, seed=42)
    assert np.allclose(angles["solar_zenith"], again["solar_zenith"])


def test_create_placeholder_geolocation_dataframe():
    """Placeholder geolocation should match RAD fill-value conventions."""
    result = geolocation.create_placeholder_geolocation_dataframe(3)

    assert list(result.columns) == ["lat", "lon", "alt"]
    assert len(result) == 3
    assert np.all(result["lat"].to_numpy() == np.float32(-999))
    assert np.all(result["lon"].to_numpy() == np.float32(-999))
    assert np.all(result["alt"].to_numpy() == np.float32(-9999))
