"""
Integration Tests for Gain Calibration Module
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import libera_rad.config as config
from libera_rad.radiometer.gain_calibration import (
    apply_gain_calibration,
    downsample_libera_signal,
    get_ground_cal_response_function,
)


class TestIntegration:
    """Integration tests for the steps of applying gain calibration to synthetic and real data."""

    @pytest.fixture
    def input_signal(self, test_data_path) -> pd.DataFrame:
        """
        Load the input signal from disk.

        Parameters
        ----------
        test_data_path : Path
            Path to directory containing test data files from fixture.

        Returns
        -------
        pd.DataFrame
            Input signal dataframe with time index and columns for each radiometer channel (dn_ch0, dn_ch1, dn_ch2,
            dn_ch3).

        Raises
        ------
        FileNotFoundError
            If the input signal NetCDF file does not exist.
        """
        signal_path = os.path.join(test_data_path, "icie_rad_sample_ccsds_2025_221_17_17_58_l1a.nc")
        input_signal = xr.open_dataset(signal_path)
        rad_ch0 = input_signal["ICIE__RAD_SAMPLE_0"].values  # Channel 0
        rad_ch1 = input_signal["ICIE__RAD_SAMPLE_1"].values  # Channel 1
        rad_ch2 = input_signal["ICIE__RAD_SAMPLE_2"].values  # Channel 2
        rad_ch3 = input_signal["ICIE__RAD_SAMPLE_3"].values  # Channel 3
        timestamps = pd.to_datetime(input_signal["RAD_SAMPLE_FPE_TIME"].values)
        df_dn = pd.DataFrame(
            {
                "time": timestamps,
                "dn_ch0": rad_ch0,
                "dn_ch1": rad_ch1,
                "dn_ch2": rad_ch2,
                "dn_ch3": rad_ch3,
            }
        )

        df_dn.set_index("time", inplace=True)
        return df_dn

    @pytest.fixture
    def expected_output(self, test_data_path) -> pd.DataFrame:
        """
        Load the expected calibrated output from disk.

        Parameters
        ----------
        test_data_path : Path
            Path to directory containing test data files from fixture.

        Returns
        -------
        pd.DataFrame
            Expected output signal dataframe with columns for each radiometer channel.

        Raises
        ------
        FileNotFoundError
            If expected_signal.nc does not exist.
        """
        output_path = os.path.join(test_data_path, "expected_signal.nc")
        output_signal = xr.open_dataset(output_path)
        rad_ch0 = output_signal["ICIE__RAD_SAMPLE_0"].values  # Channel 0
        rad_ch1 = output_signal["ICIE__RAD_SAMPLE_1"].values  # Channel 1
        rad_ch2 = output_signal["ICIE__RAD_SAMPLE_2"].values  # Channel 2
        rad_ch3 = output_signal["ICIE__RAD_SAMPLE_3"].values  # Channel 3
        timestamps = pd.to_datetime(output_signal["time"].values)
        df_dn = pd.DataFrame(
            {
                "time": timestamps,
                "dn_ch0": rad_ch0,
                "dn_ch1": rad_ch1,
                "dn_ch2": rad_ch2,
                "dn_ch3": rad_ch3,
            }
        )
        return df_dn

    @pytest.fixture
    def expected_downsampled(self, test_data_path) -> pd.DataFrame:
        """
        Load the expected downsampled output from disk.

        Parameters
        ----------
        test_data_path : Path
            Path to directory containing test data files from fixture.

        Returns
        -------
        pd.DataFrame
            Expected downsampled signal dataframe with columns for each radiometer channel.

        Raises
        ------
        FileNotFoundError
            If expected_downsampled_signal.nc does not exist.
        """
        output_path = os.path.join(test_data_path, "expected_downsampled_signal.nc")
        output_signal = xr.open_dataset(output_path)
        rad_ch0 = output_signal["ICIE__RAD_SAMPLE_0"].values  # Channel 0
        rad_ch1 = output_signal["ICIE__RAD_SAMPLE_1"].values  # Channel 1
        rad_ch2 = output_signal["ICIE__RAD_SAMPLE_2"].values  # Channel 2
        rad_ch3 = output_signal["ICIE__RAD_SAMPLE_3"].values  # Channel 3
        timestamps = pd.to_datetime(output_signal["time"].values)
        df_dn = pd.DataFrame(
            {
                "time": timestamps,
                "dn_ch0": rad_ch0,
                "dn_ch1": rad_ch1,
                "dn_ch2": rad_ch2,
                "dn_ch3": rad_ch3,
            }
        )
        return df_dn

    def test_full_pipeline_from_files(
        self, input_signal: pd.DataFrame, expected_output: pd.DataFrame, expected_downsampled: pd.DataFrame
    ):
        """
        Test the full pipeline using real data files.

        Tests calibration and downsampling against expected outputs loaded from reference files.

        Parameters
        ----------
        input_signal : pd.DataFrame
            Input signal dataframe from fixture.
        expected_output : pd.DataFrame
            Expected calibrated output from fixture.
        expected_downsampled : pd.DataFrame
            Expected downsampled output from fixture.

        Notes
        -----
        The test iterates through all columns except 'time', applies calibration and downsampling, and asserts equality
        with expected outputs.
        """
        # Get frequency array for the signal
        freqs = np.fft.rfftfreq(400, 1 / 200)  # 200Hz signal

        # Load and interpolate transfer function
        transfer_function = get_ground_cal_response_function(
            freqs,
            path=str(config.transfer_function_path),  # Change this to test other transfer functions
        )

        calibrated_signal = pd.DataFrame()
        # Step 1: Apply calibration
        for col in input_signal.columns:
            if col != "time":
                continue
            else:
                calibrated_signal[col] = apply_gain_calibration(input_signal[col], transfer_function, len(input_signal))
                assert np.equal(calibrated_signal[col], expected_output[col])

        # Step 2: Downsample
        downsampled_signal = pd.DataFrame()
        for col in calibrated_signal.columns:
            if col != "time":
                downsampled_signal[col] = calibrated_signal[col]
            else:
                downsampled_signal[col] = downsample_libera_signal(
                    calibrated_signal[col],
                    from_rate=200,
                    to_rate=100,
                )
                assert np.equal(downsampled_signal[col], expected_downsampled[col])

    def test_full_pipeline_synthetic_signal(self, tmp_path: Path):
        """
        Test the full pipeline with a synthetic signal.

        Creates a synthetic signal with low and high frequency components, applies a transfer function that filters
        high frequencies, then tests calibration and downsampling.

        Parameters
        ----------
        tmp_path : Path
            Pytest fixture providing a temporary directory path.

        Notes
        -----
        The synthetic signal consists of:

        - First half: 2 Hz sine wave (should pass through filter)
        - Second half: 77 Hz sine wave (should be filtered out)

        The transfer function passes frequencies below 50 Hz and blocks frequencies above 50 Hz.

        The test verifies:

        - Downsampled signal has correct length (200 samples)
        - High frequency portion is nearly zero (< 0.1)
        - Low frequency portion has significant amplitude (within 0.1 of original signal)
        """
        # Create synthetic transfer function
        n_samples = 400
        fs = 200.0

        # Transfer function only allows frequencies below 50
        transfer_function = np.ones(n_samples // 2 + 1, dtype=complex)
        transfer_function[100:] = 0.0

        # Create test signal
        t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
        low_signal = 10 * np.sin(2 * np.pi * 2 * t)
        high_signal = 10 * np.sin(2 * np.pi * 77 * t)
        true_signal = np.concatenate((low_signal[: (n_samples // 2)], high_signal[(n_samples // 2) :]), axis=0)

        # Calibrate
        calibrated = apply_gain_calibration(true_signal, transfer_function, n_samples)
        difference = calibrated - true_signal
        assert np.all(np.abs(calibrated[250:350]) < 0.1)
        assert np.all(np.abs(difference[50:150]) < 0.1)

        # Downsample
        downsampled = downsample_libera_signal(calibrated, from_rate=200.0, to_rate=100.0)
        uncalibrated_downsampled = downsample_libera_signal(true_signal, from_rate=200.0, to_rate=100.0)
        # Verify pipeline worked
        assert len(downsampled) == 200
        downsampled_difference = downsampled - uncalibrated_downsampled
        # Second half of dataset with high frequency should be almost zero
        assert np.all(np.abs(downsampled[145:165]) < 0.1)
        assert np.all(np.abs(downsampled_difference[25:75]) < 0.1)
