import numpy as np
import pandas as pd
import pytest

# Local
from libera_rad.calibration.constants import ChannelName
from libera_rad.radiometer.radiance import (
    calculate_numerical_nanowatts_per_dn,
    calculate_physical_nanowatts_per_dn,
    calculate_radiance,
    calculate_temperature_from_dn,
)


@pytest.mark.parametrize(
    ("channel_name", "pwm_data_name", "dave_radiance_name"),
    [
        (ChannelName.SHORTWAVE, "sci_rad_pwm0_dn", "l0_sw"),
        (ChannelName.TOTAL, "sci_rad_pwm1_dn", "l1_to"),
        (ChannelName.LONGWAVE, "sci_rad_pwm2_dn", "l2_lw"),
        (ChannelName.SPLIT_SHORTWAVE, "sci_rad_pwm3_dn", "l3_ss"),
    ],
)
def test_numerical_radiance_to_dave_results(
    ground_data, calibration_data, channel_name, pwm_data_name, dave_radiance_name
):
    """Test the conversion of radiance is done the same as Dave's IDL code results"""

    # Convert radiance from pwm signal
    calculated_radiance = calculate_radiance(
        ground_data[pwm_data_name],
        ground_data["sci_rad_bench_temp_dn"],
        channel_name=channel_name,
        calibration_data=calibration_data,
    )
    # Load Dave's data
    dave_results = ground_data[dave_radiance_name]

    # Compare the results
    pd.testing.assert_series_equal(
        dave_results, calculated_radiance, check_dtype=False, check_exact=False, check_names=False
    )


@pytest.mark.parametrize(
    ("channel_name", "pwm_data_name"),
    [
        (ChannelName.SHORTWAVE, "sci_rad_pwm0_dn"),
        (ChannelName.TOTAL, "sci_rad_pwm1_dn"),
        (ChannelName.LONGWAVE, "sci_rad_pwm2_dn"),
        (ChannelName.SPLIT_SHORTWAVE, "sci_rad_pwm3_dn"),
    ],
)
def test_nw_per_dn_physical_vs_numerical(ground_data, calibration_data, channel_name, pwm_data_name):
    """Test that the nano-watts per dn conversion is the same in the numerical as the physical conversion"""
    numerical_nw = calculate_numerical_nanowatts_per_dn(
        ground_data[pwm_data_name],
        np.median(ground_data["sci_rad_bench_temp_dn"]),
        channel_name=channel_name,
        calibration_data=calibration_data,
    )

    temp_calibration = calibration_data.housekeeping_temperature_coefficients["bench_coefficients"]
    bench_temp = calculate_temperature_from_dn(np.median(ground_data["sci_rad_bench_temp_dn"]), temp_calibration)

    physical_nw = calculate_physical_nanowatts_per_dn(
        ground_data[pwm_data_name], bench_temp, channel_name=channel_name, ground_calibration_info=calibration_data
    )

    # Compare the results as a relative error to the physical answer
    # (Goal is to be within the 1e-5, 10 ppm, range at launch) Requires a rerun on Dave's optimization system.
    assert np.abs(numerical_nw - physical_nw) / physical_nw < 1e-4
