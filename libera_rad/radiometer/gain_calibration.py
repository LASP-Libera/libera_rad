"""
Transient Gain Calibration Module for Radiometer Data

This module provides functions to:
1. Save and load gain transfer functions for radiometer channels
2. Apply gain calibration to radiometer signals using Fourier transforms
3. Downsample signals
"""

import numpy as np
import xarray as xr
from scipy import signal
from scipy.fft import irfft, rfft

from libera_rad import config


def get_ground_cal_response_function(
    freqs: np.ndarray,
    path: str = config.transfer_function_path
) -> np.ndarray:
    """
    Load and interpolate a transfer function from a NetCDF file.

    Reads a pre-computed transfer function from disk and interpolates it onto the requested frequency grid.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies (in Hz) at which to evaluate the transfer function.
    path : str, optional
        Path to the NetCDF file containing the transfer function data.
        Default is defined in the module configuration file.

    Returns
    -------
    np.ndarray
        Transfer function values interpolated at the requested frequencies.
        Same shape as the input `freqs` array.

    Notes
    -----
    The NetCDF file is expected to contain:

    - 'transfer' : The transfer function values
    - 'freq' : The corresponding frequency values
    """
    transfer_function_data = xr.open_dataset(path)
    transfer_function = transfer_function_data['transfer']
    transfer_function_freqs = transfer_function_data['freq']
    interp_transfer = np.interp(freqs, transfer_function_freqs, transfer_function)
    return interp_transfer.astype(complex)


def downsample_libera_signal(
    signal_data: np.ndarray,
    from_rate: float = 200.0,
    to_rate: float = 100.0,
    method: str = 'decimate'
) -> np.ndarray:
    """
    Downsample a signal from one sampling rate to another.

    Parameters
    ----------
    signal_data : np.ndarray
        Input signal to be downsampled. Should be a 1D array.
    from_rate : float, optional
        Original sampling rate in Hz. Default is 200.0.
    to_rate : float, optional
        Target sampling rate in Hz. Default is 100.0.
    method : str, optional
        Downsampling method to use. Options are:
        - 'decimate' : Uses scipy.signal.decimate with IIR filtering and zero-phase processing (default)
        - Any other value : Uses scipy.signal.resample for Fourier-domain resampling

    Returns
    -------
    np.ndarray
        Downsampled signal at the target sampling rate.

    Notes
    -----
    TODO: LIBSDC-710 Compare this method to the downsampling method used by the IE team.
    """
    factor = int(from_rate / to_rate)
    if len(signal_data) == 0:
        return signal_data
    if method == 'decimate':
        return signal.decimate(signal_data, factor, ftype='iir', zero_phase=True)
    else:
        n_samples_new = int(len(signal_data) * to_rate / from_rate)
        return signal.resample(signal_data, n_samples_new)


def apply_gain_calibration(
    signal_data: np.ndarray,
    transfer_function: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Apply gain calibration to a signal using the FFT method.

    Parameters
    ----------
    signal_data : np.ndarray
        Input signal to calibrate. Should be a 1D time-domain signal.
    transfer_function : np.ndarray
        Complex transfer function of the detector system. Must have length equal to (n_samples//2 + 1) to match the
        output of rfft. This represents H(f) where the measured signal = H(f) * true_signal.
    n_samples : int
        Number of samples in the original time-domain signal. Used to ensure correct inverse FFT output length.

    Returns
    -------
    np.ndarray
        Calibrated signal in the time domain with the same length as the input signal (n_samples).

    Notes
    -----
    The calibration process:
    1. Transform signal to frequency domain using real FFT
    2. Multiply by the transfer function: S_cal(f) = S_meas(f) * H(f)
    3. Transform back to time domain using inverse real FFT

    This deconvolves the detector response from the measured signal, recovering an estimate of the true
    input signal before detector effects.
    """
    # Take FFT of input signal
    signal_fft = rfft(signal_data)

    # Apply inverse of transfer function to compensate for detector response
    calibrated_fft = signal_fft * transfer_function

    # Convert back to time domain
    calibrated_signal = irfft(calibrated_fft, n=n_samples)

    return calibrated_signal
