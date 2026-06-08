import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_rad.calibration.constants import BoardName, ChannelName, DetectorTracePath
from libera_rad.config import l1b_ground_calibration_path

# Local
from libera_rad.radiometer.radiance import (
    GeometricResistance,
    _load_calibration_data,
    calculate_geometric_resistance,
    calculate_heater_current,
    calculate_heater_max_power,
    calculate_numerical_nanowatts_per_dn,
    calculate_physical_nanowatts_per_dn,
    calculate_radiance_from_dn,
    calculate_radiances,
    calculate_resistance_from_temp,
    calculate_temperature_from_dn,
    calculate_thermistor_power,
    calibrate_and_downsample_radiometer_data,
    create_emitted_power_interpolation,
    interpolate_temperatures,
)


@pytest.mark.parametrize(
    ("channel_name", "measurement_pwm", "dark_pwm", "temp_dn"),
    [
        (ChannelName.TOTAL, 8613.90, 8600.05, 1905),
        (ChannelName.SHORTWAVE, 10092.99, 10092.84, 1905),
        (ChannelName.LONGWAVE, 10013.79, 10015.23, 1905),
        (ChannelName.SPLIT_SHORTWAVE, 8456.12, 8456.04, 1905),
    ],
)
def test_physical_to_numerical_radiance(calibration_data, channel_name, measurement_pwm, dark_pwm, temp_dn):
    """
    Test the conversion of radiance is done the same for a numerical approach and a physical approach
    """
    numerical_nw = calculate_numerical_nanowatts_per_dn(
        dark_pwm, temp_dn, channel_name=channel_name, calibration_data=calibration_data
    )

    temp_calibration = calibration_data.housekeeping_temperature_coefficients["bench_coefficients"]
    bench_temp = calculate_temperature_from_dn(temp_dn, temp_calibration)

    physical_nw = calculate_physical_nanowatts_per_dn(
        dark_pwm, bench_temp, channel_name=channel_name, ground_calibration_info=calibration_data
    )

    numerical_rad = calculate_radiance_from_dn(measurement_pwm, numerical_nw, channel_name, calibration_data)
    physical_rad = calculate_radiance_from_dn(measurement_pwm, physical_nw, channel_name, calibration_data)

    # Check the relative error is less than .01%
    assert np.abs(numerical_rad - physical_rad) / numerical_rad < 1e-4


@pytest.mark.parametrize(
    ("channel", "pwm_dn", "nw_per_dn", "known_radiance"),
    [
        (ChannelName.TOTAL, 8613.90, 32.09785, 2232.904),
        (ChannelName.SHORTWAVE, 10092.99, 31.9849, 2606.272),
        (ChannelName.LONGWAVE, 10013.79, 32.8347, 2678.516),
        (ChannelName.SPLIT_SHORTWAVE, 8456.12, 32.66131, 2240.222),
    ],
)
def test_calculate_radiance_from_dn(calibration_data, channel, pwm_dn, nw_per_dn, known_radiance):
    radiance = calculate_radiance_from_dn(pwm_dn, nw_per_dn, channel, calibration_data)

    assert np.abs(radiance - known_radiance) < 1e-2


@pytest.mark.parametrize(
    ("dark_pwm", "channel_name", "heat_sink_temp_dn", "known_nw_per_dn"),
    [
        (8600.05, ChannelName.TOTAL, 1905, 32.0978521),
        (10092.84, ChannelName.SHORTWAVE, 1905, 31.9848772),
        (10015.23, ChannelName.LONGWAVE, 1905, 32.8347574),
        (8456.04, ChannelName.SPLIT_SHORTWAVE, 1905, 32.6613146),
    ],
)
def test_calculate_numerical_nanowatts_per_dn(
    calibration_data, dark_pwm, channel_name, heat_sink_temp_dn, known_nw_per_dn
):
    """
    Test the calculate_nanowatts_per_dn function
    Taken from Dave's calculations
    """

    calculated_nw_per_dn = calculate_numerical_nanowatts_per_dn(
        dark_pwm, heat_sink_temp_dn, channel_name, calibration_data
    )
    assert np.abs(calculated_nw_per_dn - known_nw_per_dn) < 1e-5


@pytest.mark.parametrize(
    ("temperature_dn", "known_temperature"), [(500, 85.52676), (1000, 63.17709), (1905, 34.17058), (2000, 31.54232)]
)
def test_calculate_temperature_from_dn(calibration_data, temperature_dn, known_temperature):
    temp_coefficients = calibration_data.housekeeping_temperature_coefficients["bench_coefficients"]
    temperature = calculate_temperature_from_dn(temperature_dn, temp_coefficients)
    assert np.abs(temperature - known_temperature) < 1e-5


@pytest.mark.parametrize(
    ("active_pwm", "channel_name", "heat_sink_temp", "known_nw_per_dn"),
    [
        (5000, ChannelName.TOTAL, 22, 32.0898),
        (5000, ChannelName.SHORTWAVE, 22, 31.9766),
        (5000, ChannelName.LONGWAVE, 22, 32.8258),
        (5000, ChannelName.SPLIT_SHORTWAVE, 22, 32.6533),
    ],
)
def test_watts_per_dn_from_physics(calibration_data, active_pwm, channel_name, heat_sink_temp, known_nw_per_dn):
    """
    Test the calculate_l1b_physical_power function.
    The known nw_per_dn are taken from Dave's calculations
    """
    physics_nw_per_dn = calculate_physical_nanowatts_per_dn(active_pwm, heat_sink_temp, channel_name, calibration_data)

    assert np.abs(physics_nw_per_dn - known_nw_per_dn) < 1e-3


@pytest.mark.parametrize(
    ("channel", "trace_path", "temperature", "resistance"),
    [
        (ChannelName.TOTAL, DetectorTracePath.HEATER, 35, 14148.65372),
        (ChannelName.TOTAL, DetectorTracePath.HEATER, 50, 14822.19272),
        (ChannelName.TOTAL, DetectorTracePath.HEATER, 65, 15495.73172),
        (ChannelName.TOTAL, DetectorTracePath.HEATER, 80, 16169.27072),
        (ChannelName.TOTAL, DetectorTracePath.THERMISTOR, 35, 5999.93004),
        (ChannelName.TOTAL, DetectorTracePath.THERMISTOR, 50, 6285.65919),
        (ChannelName.TOTAL, DetectorTracePath.THERMISTOR, 65, 6571.38834),
        (ChannelName.TOTAL, DetectorTracePath.THERMISTOR, 80, 6857.11749),
    ],
)
def test_calculate_resistance_from_temp(calibration_data, channel, trace_path, temperature, resistance):
    """
    Test the calculate_resistance_from_temp function
    """
    channel_calibration = calibration_data.channels[channel.value]
    resistance_calculated = calculate_resistance_from_temp(temperature, channel_calibration, trace_path=trace_path)
    assert np.abs(resistance_calculated - resistance) < 0.0001


@pytest.mark.parametrize(
    ("channel", "trace_path", "temperature", "known_results"),
    [
        (
            ChannelName.TOTAL,
            DetectorTracePath.HEATER,
            35,
            GeometricResistance(
                silicone_resistance=40.18218, leg_resistance=60.83921, detector_resistance=14047.7738189
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.HEATER,
            50,
            GeometricResistance(
                silicone_resistance=40.182176, leg_resistance=62.2873198, detector_resistance=14716.510486
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.HEATER,
            65,
            GeometricResistance(
                silicone_resistance=40.1821765, leg_resistance=63.7354286, detector_resistance=15385.247152
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.HEATER,
            80,
            GeometricResistance(
                silicone_resistance=40.182176, leg_resistance=65.1835375, detector_resistance=16053.9838197
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.THERMISTOR,
            35,
            GeometricResistance(
                silicone_resistance=37.979557, leg_resistance=62.279273, detector_resistance=5899.671209
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.THERMISTOR,
            50,
            GeometricResistance(
                silicone_resistance=37.979557, leg_resistance=63.762208, detector_resistance=6180.625824
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.THERMISTOR,
            65,
            GeometricResistance(
                silicone_resistance=37.979557, leg_resistance=65.245142, detector_resistance=6461.580441
            ),
        ),
        (
            ChannelName.TOTAL,
            DetectorTracePath.THERMISTOR,
            80,
            GeometricResistance(silicone_resistance=37.979557, leg_resistance=66.72807, detector_resistance=6742.53505),
        ),
    ],
)
def test_calculate_geometric_resistances(calibration_data, channel, trace_path, temperature, known_results):
    """
    Test the calculate_geometric_resistance function
    """
    channel_calibration = calibration_data.channels[channel.value]
    heat_sink_temperature = 35
    trace_resistance = calculate_resistance_from_temp(temperature, channel_calibration, trace_path=trace_path)
    heat_sink_resistance = calculate_resistance_from_temp(
        heat_sink_temperature, channel_calibration, trace_path=trace_path
    )

    calculated_results = calculate_geometric_resistance(
        trace_resistance,
        heat_sink_resistance,
        channel_specific_calibration_data=channel_calibration,
        trace_path=trace_path,
    )
    assert np.abs(calculated_results.silicone_resistance - known_results.silicone_resistance) < 0.0001
    assert np.abs(calculated_results.leg_resistance - known_results.leg_resistance) < 0.0001
    assert np.abs(calculated_results.detector_resistance - known_results.detector_resistance) < 0.0001


@pytest.mark.parametrize(
    ("channel", "temperature", "known_current"),
    [
        (ChannelName.TOTAL, 35, 1.685589e-4),
        (ChannelName.TOTAL, 50, 1.648562e-4),
        (ChannelName.TOTAL, 65, 1.613127e-4),
        (ChannelName.TOTAL, 80, 1.579183e-4),
        (ChannelName.SHORTWAVE, 35, 1.683506e-4),
        (ChannelName.SHORTWAVE, 50, 1.646507e-4),
        (ChannelName.SHORTWAVE, 65, 1.611099e-4),
        (ChannelName.SHORTWAVE, 80, 1.577182e-4),
    ],
)
def test_calculate_heater_current(calibration_data, channel, temperature, known_current):
    """
    Test the calculate_heater_current function
    """
    trace_path = DetectorTracePath.HEATER
    channel_calibration = calibration_data.channels[channel.value]
    board_calibration = calibration_data.boards[BoardName.EMFPE.value]
    heat_sink_temperature = 35
    trace_resistance = calculate_resistance_from_temp(temperature, channel_calibration, trace_path=trace_path)
    heat_sink_resistance = calculate_resistance_from_temp(
        heat_sink_temperature, channel_calibration, trace_path=trace_path
    )

    geometric_resistances = calculate_geometric_resistance(
        trace_resistance,
        heat_sink_resistance,
        channel_specific_calibration_data=channel_calibration,
        trace_path=trace_path,
    )
    current = calculate_heater_current(
        geometric_resistances,
        channel_specific_calibration_data=channel_calibration,
        board_specific_calibration_data=board_calibration,
    )
    assert np.abs(current - known_current) < 1e-8


@pytest.mark.parametrize(
    ("channel", "temperature", "known_power"),
    [
        (ChannelName.SHORTWAVE, 35, 3.9847e-4),
        (ChannelName.TOTAL, 35, 3.99989e-4),
        (ChannelName.LONGWAVE, 35, 4.09239e-4),
        (ChannelName.SPLIT_SHORTWAVE, 35, 4.069251e-4),
    ],
)
def test_calculate_heater_max_power(calibration_data, channel, temperature, known_power):
    """
    Test the calculate_heater_mean_power function
    """
    channel_calibration = calibration_data.channels[channel.value]
    board_calibration = calibration_data.boards[BoardName.EMFPE.value]
    heat_sink_temperature = 35
    mean_power = calculate_heater_max_power(
        temperature,
        heat_sink_temperature,
        channel_specific_calibration_data=channel_calibration,
        board_specific_calibration_data=board_calibration,
    )
    assert np.abs(mean_power - known_power) < 1e-7


@pytest.mark.parametrize(
    ("channel", "temperature", "known_power"),
    [
        (ChannelName.TOTAL, 35, 8.718118e-4),
        (ChannelName.TOTAL, 50, 8.734018e-4),
        (ChannelName.TOTAL, 65, 8.740449e-4),
        (ChannelName.TOTAL, 80, 8.738557e-4),
    ],
)
def test_calculate_thermistor_power(calibration_data, channel, temperature, known_power):
    """
    Test the calculate_thermistor_power function
    """
    heat_sink_temperature = 35
    channel_calibration = calibration_data.channels[channel.value]
    board_calibration = calibration_data.boards[BoardName.EMFPE.value]

    thermistor_power = calculate_thermistor_power(
        temperature,
        heat_sink_temperature,
        channel_specific_calibration_data=channel_calibration,
        board_specific_calibration_data=board_calibration,
    )
    assert np.abs(thermistor_power - known_power) < 1e-8


@pytest.mark.parametrize(
    ("channel", "emitted_power_first", "emitted_power_last"),
    [
        (ChannelName.TOTAL, 0, 1.2606372e-2),
        (ChannelName.SHORTWAVE, 0, 1.1952259e-2),
        (ChannelName.LONGWAVE, 0, 1.3038966733e-2),
        (ChannelName.SPLIT_SHORTWAVE, 0, 1.20764668e-2),
    ],
)
def test_create_emitted_power_interpolation(calibration_data, channel, emitted_power_first, emitted_power_last):
    """
    Test the create_emitted_power_interpolation function
    """
    channel_calibration = calibration_data.channels[channel.value]
    heat_sink_temp = 35
    temp_range = np.arange(heat_sink_temp, heat_sink_temp + 200, 0.1) + 273.15
    emitted_power_range = create_emitted_power_interpolation(
        temp_range, heat_sink_temp, channel_specific_calibration_data=channel_calibration
    )
    assert np.abs(emitted_power_range[0] - emitted_power_first) < 1e-2
    assert np.abs(emitted_power_range[-1] - emitted_power_last) < 1e-2
    assert len(emitted_power_range) == 2000


class TestLoadCalibrationData:
    """Tests for _load_calibration_data function."""

    def test_load_calibration_data_success(self):
        """Test successful loading of calibration data."""
        calibration_data = _load_calibration_data()
        expected = json.load(open(l1b_ground_calibration_path))
        for channel in calibration_data.channels:
            assert channel in expected["channels"]
        for board in calibration_data.boards:
            assert board in expected["boards"]
        for coefficients in calibration_data.housekeeping_temperature_coefficients:
            assert coefficients in expected["housekeeping_temperature_coefficients"]

    def test_load_calibration_data_file_not_found(self):
        """Test error when calibration file is not found."""
        with pytest.raises(FileNotFoundError, match="Calibration file not found"):
            _load_calibration_data(Path("not/a/path"))


class TestCalibrateAndDownsampleRadiometerData:
    """Tests for _calibrate_and_downsample_radiometer_data function."""

    @pytest.fixture
    def mock_rad_data(self):
        """Create mock radiometer dataset."""
        times = pd.date_range("2025-01-01", periods=1000, freq="ms").values
        return xr.Dataset(
            {
                "RAD_SAMPLE_FPE_TIME": (["time"], times),
                "ICIE__RAD_SAMPLE_0": (["time"], np.random.rand(1000)),
                "ICIE__RAD_SAMPLE_1": (["time"], np.random.rand(1000)),
                "ICIE__RAD_SAMPLE_2": (["time"], np.random.rand(1000)),
                "ICIE__RAD_SAMPLE_3": (["time"], np.random.rand(1000)),
            }
        )

    @pytest.fixture
    def mock_rad_data_missing_channel(self):
        """Create mock radiometer dataset."""
        times = pd.date_range("2025-01-01", periods=1000, freq="ms").values
        return xr.Dataset(
            {
                "RAD_SAMPLE_FPE_TIME": (["time"], times),
                "ICIE__RAD_SAMPLE_1": (["time"], np.random.rand(1000)),
                "ICIE__RAD_SAMPLE_2": (["time"], np.random.rand(1000)),
                "ICIE__RAD_SAMPLE_3": (["time"], np.random.rand(1000)),
            }
        )

    def test_calibrate_and_downsample_success(self, mock_rad_data):
        """Test successful calibration and downsampling."""
        with (
            patch("libera_rad.radiometer.gain_calibration.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
        ):
            mock_downsample.side_effect = lambda x: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)

            timestamps, calibrated_data = calibrate_and_downsample_radiometer_data(mock_rad_data)

            assert len(timestamps) == 500  # Downsampled from 1000
            assert "total" in calibrated_data
            assert "lw" in calibrated_data

    def test_calibrate_and_downsample_missing_channel(self, mock_rad_data_missing_channel, caplog):
        """Test warning when channel variable is not found."""

        with (
            patch("libera_rad.radiometer.gain_calibration.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
        ):
            mock_downsample.side_effect = lambda x: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)

            timestamps, calibrated_data = calibrate_and_downsample_radiometer_data(mock_rad_data_missing_channel)

            assert "sw" not in calibrated_data
            assert "No variable found for channel sw" in caplog.text


class TestInterpolateTemperatures:
    """Tests for interpolate_temperatures function."""

    def test_interpolate_temperatures(self):
        """Test temperature interpolation."""
        base = np.datetime64("2025-01-01T00:00:00")
        timestamps = base + np.array([100, 200, 300, 400, 500], dtype="timedelta64[ms]")

        nom_hk_data = xr.Dataset(
            {
                "PACKET_ICIE_TIME": (["time"], base + np.array([0, 250, 500], dtype="timedelta64[ms]")),
                "ICIE__FPE_TSCOPE_TEMP": (["time"], np.array([20.0, 25.0, 30.0])),
            }
        )

        result = interpolate_temperatures(timestamps, nom_hk_data)
        expected = np.array([22.0, 24.0, 26.0, 28.0, 30.0])
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=0, atol=1e-5)


class TestCalculateRadiances:
    """Tests for calculate_radiances function."""

    def test_calculate_radiances_success(self):
        """Test radiance calculation."""
        calibrated_data = {"sw": np.random.rand(100), "lw": np.random.rand(100)}
        temperatures = pd.Series(np.full(100, 25.0))

        with (
            patch("libera_rad.calibration.constants.get_channel_name_enum") as mock_get_enum,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_calc_rad,
        ):
            mock_get_enum.side_effect = [ChannelName.SHORTWAVE, ChannelName.LONGWAVE]
            mock_calc_rad.return_value = np.random.rand(100)

            result = calculate_radiances(calibrated_data, temperatures)

            assert "sw" in result
            assert "lw" in result
            assert len(result["sw"]) == 100

    def test_calculate_radiances_invalid_channel(self, caplog):
        """Test handling of invalid channel names."""
        calibrated_data = {"invalid": np.random.rand(100)}
        temperatures = pd.Series(np.full(100, 25.0))

        with patch("libera_rad.calibration.constants.get_channel_name_enum", return_value=None):
            result = calculate_radiances(calibrated_data, temperatures)

            assert "invalid" not in result
            assert "Could not convert channel string 'invalid' to enum" in caplog.text
