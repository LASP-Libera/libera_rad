"""Unit tests for geolocation helpers."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from libera_rad import geolocation


def test_spacecraft_ecef_positions():
    u_gps_times = np.array([1.0, 2.0])
    mock_instrument = object()
    with (
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1.0, 2.0])),
        patch("libera_rad.geolocation.sp.obj.Body", return_value=mock_instrument),
        patch("libera_rad.geolocation.spatial.SpatialQueries.query_rotation_and_position") as mock_query,
    ):
        mock_query.side_effect = [
            ((np.eye(3), np.array([1.0, 2.0, 3.0])), 0),
            ((np.eye(3), np.array([4.0, 5.0, 6.0])), 0),
        ]
        result = geolocation._spacecraft_ecef_positions(u_gps_times)

    assert mock_query.call_count == 2
    assert list(result.columns) == ["x", "y", "z"]
    assert np.allclose(result.iloc[0].to_numpy(), [1.0, 2.0, 3.0])
    assert np.allclose(result.iloc[1].to_numpy(), [4.0, 5.0, 6.0])


def test_subsatellite_lla_from_ecef():
    sc_xyz_df = pd.DataFrame({"x": [6378.0], "y": [0.0], "z": [0.0]})
    with patch("libera_rad.geolocation.spatial.ecef_to_geodetic") as mock_ecef:
        mock_ecef.return_value = np.array([[10.0, 20.0, 0.1]])
        result = geolocation._subsatellite_lla_from_ecef(sc_xyz_df)

    mock_ecef.assert_called_once()
    assert list(result.columns) == ["lat", "lon", "alt"]
    assert result["lon"].iloc[0] == 10.0
    assert result["lat"].iloc[0] == 20.0
    assert result["alt"].iloc[0] == 0.1


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
