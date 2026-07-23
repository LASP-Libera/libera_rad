"""Unit tests for geolocation helpers."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from spiceypy.utils.exceptions import SpiceyError

from libera_rad import geolocation


def test_calculate_geometry_uses_curryer():
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    km = Mock()
    fields_df = pd.DataFrame(
        {
            "subsatellite_latitude": [10.0, 11.0],
            "subsatellite_longitude": [20.0, 21.0],
            "subsatellite_colatitude": [80.0, 79.0],
            "subsolar_latitude": [-5.0, -4.0],
            "subsolar_longitude": [100.0, 101.0],
            "subsolar_colatitude": [95.0, 94.0],
            "spacecraft_radius": [7000.0, 7001.0],
            "spacecraft_altitude": [800.0, 801.0],
        }
    )
    mock_geo = Mock()
    mock_geo.get_geometry.return_value = fields_df
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo) as mock_cls,
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1, 2])),
    ):
        result = geolocation.calculate_geometry(km, timestamps)

    km.ensure_known_kernels_are_furnished.assert_called_once()
    mock_cls.assert_called_once_with("JPSS4_SC")
    _, kwargs = mock_geo.get_geometry.call_args
    assert kwargs["fields"] == list(geolocation._GEOMETRY_FIELDS)
    assert "spacecraft_altitude" in result.columns


def test_calculate_geometry_raises_friendly_message_on_spice_error():
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    mock_geo = Mock()
    mock_geo.get_geometry.side_effect = SpiceyError("SPICE(NOFRAMECONNECT)")
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
        patch("libera_rad.geolocation._spice_error_message", return_value="no coverage for the requested time"),
    ):
        with pytest.raises(RuntimeError, match="no coverage for the requested time"):
            geolocation.calculate_geometry(km, timestamps)


def _all_nan_geometry():
    return pd.DataFrame({column: [np.nan] for field in geolocation._GEOMETRY_FIELDS for column in field.columns})


def test_calculate_geometry_raises_when_no_coverage():
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    mock_geo = Mock()
    mock_geo.get_geometry.return_value = _all_nan_geometry()
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
    ):
        with pytest.raises(RuntimeError, match="no coverage"):
            geolocation.calculate_geometry(km, timestamps)


def test_calculate_geometry_allows_all_nan_when_coverage_not_required():
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    mock_geo = Mock()
    mock_geo.get_geometry.return_value = _all_nan_geometry()
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
    ):
        result = geolocation.calculate_geometry(km, timestamps, require_coverage=False)
    assert result.isna().to_numpy().all()


def test_create_placeholder_geometry():
    result = geolocation.create_placeholder_geometry(5)
    assert len(result) == 5
    expected_columns = [column for field in geolocation._GEOMETRY_FIELDS for column in field.columns]
    assert list(result.columns) == expected_columns
    assert np.all(result["subsatellite_latitude"].to_numpy() == np.float32(-999))
    assert np.all(result["spacecraft_radius"].to_numpy() == np.float64(-9999))


def test_subsatellite_lat_lon_alt():
    geometry_data = pd.DataFrame(
        {
            "subsatellite_latitude": [10.0, 11.0],
            "subsatellite_longitude": [20.0, 21.0],
            "spacecraft_altitude": [800.0, 801.0],
        }
    )
    result = geolocation.subsatellite_lat_lon_alt(geometry_data)
    assert list(result.columns) == ["lat", "lon", "alt"]
    assert result["lat"].tolist() == [10.0, 11.0]
    assert result["lon"].tolist() == [20.0, 21.0]
    assert result["alt"].tolist() == [800.0, 801.0]


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


def test_create_placeholder_geolocation_dataframe():
    """Placeholder geolocation should match RAD fill-value conventions."""
    result = geolocation.create_placeholder_geolocation_dataframe(3)

    assert list(result.columns) == ["lat", "lon", "alt"]
    assert len(result) == 3
    assert np.all(result["lat"].to_numpy() == np.float32(-999))
    assert np.all(result["lon"].to_numpy() == np.float32(-999))
    assert np.all(result["alt"].to_numpy() == np.float32(-9999))


def test_az_el_from_et_returns_none_on_spice_error():
    with patch("libera_rad.geolocation.sp.pxform", side_effect=SpiceyError("SPICE(NOFRAME)")):
        assert geolocation._az_el_from_et(0.0) is None


def test_calculate_azimuth_elevation_for_timestamps():
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    et_times = np.array([1.0, 2.0])
    km = Mock()
    with (
        patch.object(
            geolocation, "_az_el_on_et_times", return_value=(np.zeros(2, np.float32), np.zeros(2, np.float32))
        ) as mock_full,
        patch("libera_rad.geolocation.spicetime.adapt", return_value=et_times),
    ):
        geolocation.calculate_azimuth_elevation_for_timestamps(km, timestamps)

    km.ensure_known_kernels_are_furnished.assert_called_once()
    mock_full.assert_called_once_with(et_times, -999.0)
