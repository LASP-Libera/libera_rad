"""Unit tests for geolocation helpers."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from spiceypy.utils.exceptions import SpiceyError

from libera_rad import constants, geolocation


def test_calculate_geometry_uses_curryer():
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    km = Mock()
    spacecraft_df = pd.DataFrame(
        {
            "subsatellite_latitude": [10.0, 11.0],
            "subsatellite_longitude": [20.0, 21.0],
            "subsatellite_colatitude": [80.0, 79.0],
            "subsolar_latitude": [-5.0, -4.0],
            "subsolar_longitude": [100.0, 101.0],
            "subsolar_colatitude": [95.0, 94.0],
            "spacecraft_radius": [7000.0, 7001.0],
            "spacecraft_altitude": [800.0, 801.0],
            "spacecraft_position_inertial_x": [5000.0, 5001.0],
            "spacecraft_position_inertial_y": [4000.0, 4001.0],
            "spacecraft_position_inertial_z": [1000.0, 1001.0],
            "spacecraft_velocity_inertial_x": [-2.9, -2.9],
            "spacecraft_velocity_inertial_y": [3.5, 3.5],
            "spacecraft_velocity_inertial_z": [6.0, 6.0],
            "attitude_q0": [1.0, 1.0],
            "attitude_q1": [0.0, 0.0],
            "attitude_q2": [0.0, 0.0],
            "attitude_q3": [0.0, 0.0],
        }
    )
    instrument_df = pd.DataFrame(
        {
            "viewing_zenith": [10.0, 11.0],
            "solar_zenith": [40.0, 41.0],
            "viewing_azimuth": [100.0, 101.0],
            "solar_azimuth": [120.0, 121.0],
            "relative_azimuth": [150.0, 151.0],
            "cone_angle": [5.0, 6.0],
            "cone_angle_rate": [0.5, -0.5],
            "clock_angle": [90.0, 270.0],
            "clock_angle_rate": [1.0, -1.0],
            "along_track_angle": [-0.16, -0.16],
            "cross_track_angle": [-30.0, 30.0],
        }
    )
    spacecraft_geo = Mock()
    spacecraft_geo.get_geometry.return_value = spacecraft_df
    instrument_geo = Mock()
    instrument_geo.get_geometry.return_value = instrument_df
    with (
        patch(
            "libera_rad.geolocation.geometry.GeometryData",
            side_effect=[spacecraft_geo, instrument_geo],
        ) as mock_cls,
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1, 2])),
    ):
        result = geolocation.calculate_geometry(km, timestamps)

    km.ensure_known_kernels_are_furnished.assert_called_once()
    assert [call.args[0] for call in mock_cls.call_args_list] == ["JPSS4_SC", "LIBERA_RAD"]
    assert spacecraft_geo.get_geometry.call_args.kwargs["fields"] == list(geolocation._SPACECRAFT_FIELDS)
    assert instrument_geo.get_geometry.call_args.kwargs["fields"] == list(geolocation._INSTRUMENT_FIELDS)
    assert "spacecraft_altitude" in result.columns
    assert "cone_angle" in result.columns
    assert "attitude_q0" in result.columns
    assert "clock_angle" in result.columns


def test_calculate_start_of_hour_state():
    # "Start of hour" is a mission timetag convention: the 24 top-of-hour epochs of the
    # granule's UTC day, queried from curryer's generic position/velocity fields.
    timestamps = np.array(["2025-11-20T17:59:50", "2025-11-20T18:00:20"], dtype="datetime64[ns]")
    km = Mock()
    hourly_df = pd.DataFrame(
        {
            "spacecraft_position_inertial_x": np.arange(24, dtype=float),
            "spacecraft_position_inertial_y": np.arange(24, dtype=float) + 100.0,
            "spacecraft_position_inertial_z": np.arange(24, dtype=float) + 200.0,
            "spacecraft_velocity_inertial_x": np.arange(24, dtype=float) + 300.0,
            "spacecraft_velocity_inertial_y": np.arange(24, dtype=float) + 400.0,
            "spacecraft_velocity_inertial_z": np.arange(24, dtype=float) + 500.0,
        }
    )
    mock_geo = Mock()
    mock_geo.get_geometry.return_value = hourly_df
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo) as mock_cls,
        patch("libera_rad.geolocation.spicetime.adapt", side_effect=lambda t, _fmt: np.arange(len(t))) as mock_adapt,
    ):
        position, velocity = geolocation.calculate_start_of_hour_state(km, timestamps)

    km.ensure_known_kernels_are_furnished.assert_called_once()
    mock_cls.assert_called_once_with("JPSS4_SC")
    assert mock_geo.get_geometry.call_args.kwargs["fields"] == ["sc_position_inertial", "sc_velocity_inertial"]
    assert position.shape == (24, 3)
    assert velocity.shape == (24, 3)
    np.testing.assert_allclose(position[:, 0], np.arange(24))
    np.testing.assert_allclose(velocity[:, 0], np.arange(24) + 300.0)

    # The queried epochs are the top of each hour of the granule's start day.
    queried = mock_adapt.call_args.args[0]
    assert len(queried) == 24
    assert list(queried.hour) == list(range(24))
    assert (queried.normalize() == pd.Timestamp("2025-11-20")).all()


def test_calculate_start_of_hour_state_rejects_unknown_observer():
    """The start-of-hour state is a spacecraft query; an instrument frame must be rejected."""
    timestamps = np.array(["2025-11-20T17:59:50"], dtype="datetime64[ns]")
    km = Mock()
    with patch("libera_rad.geolocation.geometry.GeometryData") as mock_cls:
        with pytest.raises(ValueError, match="Unsupported spacecraft observer"):
            geolocation.calculate_start_of_hour_state(km, timestamps, observer="LIBERA_RAD")
    mock_cls.assert_not_called()
    km.ensure_known_kernels_are_furnished.assert_not_called()


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


def test_calculate_geometry_raises_when_only_subsolar_covered():
    # Out of spacecraft coverage the subsolar point (Sun ephemeris only) stays finite while the
    # spacecraft-derived fields go NaN; coverage is judged by the spacecraft fields, so this must
    # still raise rather than emit a product with NaN positions.
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    spacecraft_df = pd.DataFrame(
        {column: [np.nan] for field in geolocation._SPACECRAFT_FIELDS for column in field.columns}
    )
    for column in geolocation.geometry.GeometryField.SUBSOLAR.columns:
        spacecraft_df[column] = [1.0]
    mock_geo = Mock()
    mock_geo.get_geometry.return_value = spacecraft_df
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", return_value=mock_geo),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
    ):
        with pytest.raises(RuntimeError, match="no coverage"):
            geolocation.calculate_geometry(km, timestamps)


def test_calculate_geometry_allows_all_nan_instrument_fields():
    # In jpss_only the instrument observer is legitimately all-NaN; only the spacecraft
    # observer requires coverage, so this must not raise.
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    spacecraft_geo = Mock()
    spacecraft_geo.get_geometry.return_value = pd.DataFrame(
        {column: [1.0] for field in geolocation._SPACECRAFT_FIELDS for column in field.columns}
    )
    instrument_geo = Mock()
    instrument_geo.get_geometry.return_value = pd.DataFrame(
        {column: [np.nan] for field in geolocation._INSTRUMENT_FIELDS for column in field.columns}
    )
    with (
        patch("libera_rad.geolocation.geometry.GeometryData", side_effect=[spacecraft_geo, instrument_geo]),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
    ):
        result = geolocation.calculate_geometry(km, timestamps)
    assert result["cone_angle"].isna().to_numpy().all()
    assert not result["subsatellite_latitude"].isna().to_numpy().any()


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"spacecraft_observer": "LIBERA_RAD"}, "Unsupported spacecraft observer"),
        ({"instrument_observer": "LIBERA_WFOV_CAM"}, "Unsupported instrument observer"),
        ({"instrument_observer": "JPSS4_SC"}, "Unsupported instrument observer"),
    ],
)
def test_calculate_geometry_rejects_unknown_observer(kwargs, expected_message):
    """A valid SPICE frame used in the wrong role must fail loudly, not compute the wrong optic."""
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    with patch("libera_rad.geolocation.geometry.GeometryData") as mock_cls:
        with pytest.raises(ValueError, match=expected_message):
            geolocation.calculate_geometry(km, timestamps, **kwargs)
    mock_cls.assert_not_called()
    km.ensure_known_kernels_are_furnished.assert_not_called()


@pytest.mark.parametrize("instrument_observer", constants.INSTRUMENT_OBSERVERS)
def test_calculate_geometry_accepts_every_instrument_observer(instrument_observer):
    """Every frame listed in INSTRUMENT_OBSERVERS is a valid instrument observer."""
    timestamps = np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]")
    km = Mock()
    spacecraft_geo = Mock()
    spacecraft_geo.get_geometry.return_value = pd.DataFrame(
        {column: [1.0] for field in geolocation._SPACECRAFT_FIELDS for column in field.columns}
    )
    instrument_geo = Mock()
    instrument_geo.get_geometry.return_value = pd.DataFrame(
        {column: [1.0] for field in geolocation._INSTRUMENT_FIELDS for column in field.columns}
    )
    with (
        patch(
            "libera_rad.geolocation.geometry.GeometryData",
            side_effect=[spacecraft_geo, instrument_geo],
        ) as mock_cls,
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([1])),
    ):
        geolocation.calculate_geometry(km, timestamps, instrument_observer=instrument_observer)

    assert [call.args[0] for call in mock_cls.call_args_list] == ["JPSS4_SC", instrument_observer]


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


def test_calculate_azimuth_elevation_checks_motor_ck_coverage():
    """LIBSDC-788: a motor-CK gap is reported up front, not hidden behind per-sample fill."""
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    km = Mock()
    with (
        patch.object(
            geolocation, "_az_el_on_et_times", return_value=(np.zeros(2, np.float32), np.zeros(2, np.float32))
        ),
        # First adapt() call converts to uGPS for the coverage window, second to ET.
        patch(
            "libera_rad.geolocation.spicetime.adapt",
            side_effect=[np.array([100, 300]), np.array([1.0, 2.0])],
        ),
    ):
        geolocation.calculate_azimuth_elevation_for_timestamps(km, timestamps)

    km.ensure_kernel_coverage.assert_called_once_with(geolocation.MOTOR_CK_FRAMES, 100, 300, error=False)


def test_calculate_azimuth_elevation_skips_coverage_check_when_no_samples():
    km = Mock()
    with (
        patch.object(
            geolocation, "_az_el_on_et_times", return_value=(np.zeros(0, np.float32), np.zeros(0, np.float32))
        ),
        patch("libera_rad.geolocation.spicetime.adapt", return_value=np.array([])),
    ):
        geolocation.calculate_azimuth_elevation_for_timestamps(km, np.array([], dtype="datetime64[ns]"))

    km.ensure_kernel_coverage.assert_not_called()
