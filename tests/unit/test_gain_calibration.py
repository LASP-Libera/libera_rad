"""
Unit Tests for Gain Calibration Module
"""

import numpy as np
import pytest
import xarray as xr

from libera_rad.radiometer.gain_calibration import (
    apply_gain_calibration,
    downsample_libera_signal,
    get_ground_cal_response_function,
)


class TestGetGroundCalResponseFunction:
    """Tests for transfer function loading and interpolation."""

    def test_interpolation_on_matching_frequencies(self, tmp_path):
        """Test interpolation at exact stored frequencies."""
        # Create test data
        test_freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        test_transfer = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

        # Save to NetCDF
        ds = xr.Dataset({"transfer": (["freq_dim"], test_transfer), "freq": (["freq_dim"], test_freqs)})
        file_path = tmp_path / "transfer.nc"
        ds.to_netcdf(file_path)

        # Test interpolation at exact frequencies
        result = get_ground_cal_response_function(test_freqs, path=file_path)
        np.testing.assert_array_almost_equal(result, test_transfer)

    def test_interpolation_between_frequencies(self, tmp_path):
        """Test that interpolation works correctly between known frequencies."""
        # Create test data with linear transfer function for easy verification
        test_freqs = np.array([0.0, 10.0, 20.0])
        test_transfer = np.array([1.0, 0.5, 0.0])

        ds = xr.Dataset({"transfer": (["freq_dim"], test_transfer), "freq": (["freq_dim"], test_freqs)})
        file_path = tmp_path / "transfer.nc"
        ds.to_netcdf(file_path)

        # Request interpolation at midpoints
        query_freqs = np.array([5.0, 15.0])
        expected = np.array([0.75, 0.25])  # Linear interpolation

        result = get_ground_cal_response_function(query_freqs, path=file_path)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extrapolation_behavior(self, tmp_path):
        """Test behavior when requesting frequencies outside the stored range."""
        test_freqs = np.array([10.0, 20.0, 30.0])
        test_transfer = np.array([0.9, 0.8, 0.7])

        ds = xr.Dataset({"transfer": (["freq_dim"], test_transfer), "freq": (["freq_dim"], test_freqs)})
        file_path = tmp_path / "transfer.nc"
        ds.to_netcdf(file_path)

        # np.interp extrapolates using edge values
        query_freqs = np.array([5.0, 35.0])
        result = get_ground_cal_response_function(query_freqs, path=file_path)

        # Should return edge values for out-of-range frequencies
        assert result[0] == 0.9  # Below range
        assert result[1] == 0.7  # Above range

    def test_file_not_found(self):
        """Test that appropriate error is raised when file doesn't exist."""
        with pytest.raises((FileNotFoundError, OSError)):
            get_ground_cal_response_function(np.array([1.0, 2.0]), path="nonexistent_file.nc")


class TestDownsampleSignal:
    """Tests for signal downsampling functionality."""

    def test_downsample_by_factor_of_2(self):
        """Test downsampling by factor of 2."""
        # Create a simple signal
        t = np.linspace(0, 1, 200, endpoint=False)
        signal_data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

        result = downsample_libera_signal(signal_data, from_rate=200, to_rate=100)

        # Should have half the samples
        assert len(result) == 100

        # Signal should still be smooth (no major artifacts)
        assert np.all(np.abs(result) <= 1.1)  # Allow small overshoot from filtering

    def test_downsample_by_factor_of_4(self):
        """Test downsampling by larger factor."""
        signal_data = np.random.randn(400)

        result = downsample_libera_signal(signal_data, from_rate=400, to_rate=100)

        assert len(result) == 100

    def test_downsample_non_integer_round_up(self):
        """Test downsampling with a non-integer factor, rounded up to the nearest whole number."""
        # Create a simple signal
        t = np.linspace(0, 1, 200, endpoint=False)
        signal_data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

        # 200/49 ≈ 4.08, which rounds to decimation factor 4.
        result = downsample_libera_signal(signal_data, from_rate=200, to_rate=49)

        # Factor rounds to 4, so rate = 200 / 4 = 50
        assert len(result) == 50

    def test_downsample_non_integer_round_down(self):
        """Test downsampling with a non-integer factor, rounded down to the nearest whole number"""
        # Create a simple signal
        t = np.linspace(0, 1, 200, endpoint=False)
        signal_data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

        # 200/51 ≈ 3.92, which rounds to decimation factor 4.
        result = downsample_libera_signal(signal_data, from_rate=200, to_rate=51)

        # Factor rounds to 4, so rate = 200 / 4 = 50
        assert len(result) == 50

    def test_downsample_preserves_dc_component(self):
        """Test that DC (zero frequency) component is preserved."""
        # Constant signal (DC only)
        signal_data = np.ones(200) * 5.0

        result = downsample_libera_signal(signal_data, from_rate=200, to_rate=100)

        # Mean should be approximately preserved
        np.testing.assert_almost_equal(np.mean(result), 5.0, decimal=1)

    def test_downsample_non_integer_factor(self):
        """Test downsampling with non-integer factor."""
        signal_data = np.random.randn(300)

        # 300 Hz -> 100 Hz (factor of 3)
        result = downsample_libera_signal(signal_data, from_rate=300, to_rate=100)

        assert len(result) == 100

    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        signal_data = np.random.randn(200)

        # Default: 200 Hz -> 100 Hz, decimate method
        result = downsample_libera_signal(signal_data)

        assert len(result) == 100

    def test_empty_signal(self):
        """Test behavior with empty input."""
        signal_data = np.array([])

        # This should work but return empty array
        result = downsample_libera_signal(signal_data, from_rate=200, to_rate=100)
        assert len(result) == 0

    def test_negative_from_rate(self):
        """Test behavior with negative from_rate input."""
        signal_data = np.random.randn(200)

        # Should raise ValueError
        with pytest.raises(ValueError, match="from_rate must be positive"):
            downsample_libera_signal(signal_data, from_rate=-200, to_rate=100)

    def test_negative_to_rate(self):
        """Test behavior with negative to_rate input."""
        signal_data = np.random.randn(200)

        # Should raise ValueError
        with pytest.raises(ValueError, match="to_rate must be positive"):
            downsample_libera_signal(signal_data, from_rate=200, to_rate=-100)

    def test_from_rate_less_than_to_rate(self):
        """Test behavior with to_rate higher than from_rate input."""
        signal_data = np.random.randn(200)

        # Should raise ValueError
        with pytest.raises(ValueError, match="from_rate must be greater than to_rate"):
            downsample_libera_signal(signal_data, from_rate=100, to_rate=200)


class TestApplyGainCalibration:
    """Tests for FFT-based gain calibration."""

    def test_flat_transfer_function(self):
        """Test that flat (unity) transfer function returns original signal."""
        n_samples = 100
        signal_data = np.random.randn(n_samples)

        # Unity transfer function (no detector effects)
        transfer_function = np.ones(n_samples // 2 + 1, dtype=complex)

        result = apply_gain_calibration(signal_data, transfer_function, n_samples)

        # Should recover original signal
        np.testing.assert_array_almost_equal(result, signal_data, decimal=10)

    def test_preserves_signal_length(self):
        """Test that calibration preserves signal length."""
        for n_samples in [100, 127, 256, 1000]:
            signal_data = np.random.randn(n_samples)
            transfer_function = np.ones(n_samples // 2 + 1, dtype=complex)

            result = apply_gain_calibration(signal_data, transfer_function, n_samples)

            assert len(result) == n_samples

    def test_calibration_is_linear(self):
        """Test that calibration is a linear operation."""
        n_samples = 200
        signal1 = np.random.randn(n_samples)
        signal2 = np.random.randn(n_samples)

        transfer_function = np.ones(n_samples // 2 + 1, dtype=complex) * 2.0

        # Calibrate combined signal
        result_combined = apply_gain_calibration(signal1 + signal2, transfer_function, n_samples)

        # Calibrate separately and add
        result1 = apply_gain_calibration(signal1, transfer_function, n_samples)
        result2 = apply_gain_calibration(signal2, transfer_function, n_samples)
        result_separate = result1 + result2

        # Should be equal (linearity)
        np.testing.assert_array_almost_equal(result_combined, result_separate, decimal=10)

    def test_single_sample_signal(self):
        """Test behavior with single-sample signal."""
        signal_data = np.array([1.0])
        transfer_function = np.array([1.0 + 0j])

        result = apply_gain_calibration(signal_data, transfer_function, 1)
        assert len(result) == 1

    def test_very_large_signal(self):
        """Test that functions can handle large signals efficiently."""
        n_samples = 100000
        signal_data = np.random.randn(n_samples)
        transfer_function = np.ones(n_samples // 2 + 1, dtype=complex)

        result = apply_gain_calibration(signal_data, transfer_function, n_samples)
        assert len(result) == n_samples
