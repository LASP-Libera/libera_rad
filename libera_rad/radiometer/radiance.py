"""Module for calculating the radiance of a detector for the Libera mission"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import constants

from libera_rad.calibration.calibration_models import (
    BoardCalibrations,
    ChannelCalibrations,
    LiberaGroundCalibration,
    TemperatureCoefficients,
)
from libera_rad.calibration.constants import (
    BoardName,
    ChannelName,
    DetectorTracePath,
    DetectorType,
    RadianceMethod,
    find_channel_variable,
    get_channel_name_enum,
)
from libera_rad.calibration.constants import (
    HousekeepingTemperatureCoefficient as TemperatureCoefficient,
)
from libera_rad.config import l1b_ground_calibration_path
from libera_rad.radiometer.gain_calibration import (
    apply_gain_calibration,
    decimation_factor,
    downsample_libera_signal,
    get_ground_cal_response_function,
)

_RAD_SAMPLE_HZ = 200
_L1B_OUTPUT_HZ = 100

logger = logging.getLogger(__name__)


def _load_calibration_data(calibration_data_path: Path = l1b_ground_calibration_path) -> LiberaGroundCalibration:
    """
    Load ground calibration data from JSON file.

    Reads the L1B ground calibration parameters from a JSON configuration file located in the data directory.
    Parameters
    ----------
    calibration_data_path : Path
        Path to JSON configuration file containing L1B ground calibration parameters.

    Returns
    -------
    LiberaGroundCalibration
        Calibration data object containing channel-specific calibration parameters, response functions, and other
        calibration coefficients.

    Raises
    ------
    FileNotFoundError
        If the calibration file 'l1b_ground_calibration.json' is not found in the data directory.
    json.JSONDecodeError
        If the calibration file contains invalid JSON.
    """
    if not calibration_data_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_data_path}")

    with open(calibration_data_path) as f:
        ground_calibration = json.load(f)
        return LiberaGroundCalibration(**ground_calibration)


CALIBRATION_DATA = _load_calibration_data()


def calculate_radiance(
    pwm_dn_data: float | pd.Series,
    temperature_dn_data: float | pd.Series,
    channel_name: ChannelName,
    calibration_data: LiberaGroundCalibration,
    radiance_method: RadianceMethod = RadianceMethod.NUMERICAL,
    electronics_board_name: BoardName = BoardName.EMFPE,
    temperature_coefficient_name: TemperatureCoefficient = TemperatureCoefficient.BENCH_COEFFICIENTS,
) -> float:
    """
    Calculate the radiance of a Libera detector for a single channel.

    As the physical derivation of radiance is not significantly different from the numerical derivation, we expect to
    later change this to use the physically derived method in operations, however for now to match to instrument tests
    we are using the numerical method.

    A more thorough example and explanation of the radiance calculation can be found in the Libera Radiometer
    Algorithm Theoretical Basis Document (ATBD) and the provided learning notebooks in this repository.

    Parameters
    ----------
    pwm_dn_data : Union[float, pd.Series]
        The pulse-width-modulation (PWM) data in a count format (dn) from the detector
    temperature_dn_data : Union[float, pd.Series]
        The temperature data in a count format (dn) from the detector
    channel_name : ChannelName
        The channel name of the detector. Options are SHORTWAVE, TOTAL, LONGWAVE, and SPLIT_SHORTWAVE
    calibration_data : LiberaGroundCalibration
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the LiberaGroundCalibration object
    radiance_method : RadianceMethod, optional
        The method used to calculate the radiance. Options are PHYSICAL and NUMERICAL. Default is NUMERICAL
    electronics_board_name : BoardName, optional
        The name of the electronics board used for the detector. Default is EMFPE
    temperature_coefficient_name : HousekeepingTemperatureCoefficient, optional
        The name of the temperature coefficient used for the detector. Default is BENCH_COEFFICIENTS

    Returns
    -------
    radiance : float
        The radiance of the detector in units of W m^-2 sr^-1
    """
    if radiance_method == RadianceMethod.NUMERICAL:
        nw_per_dn = calculate_numerical_nanowatts_per_dn(
            pwm_dn_data,
            temperature_dn_data,
            channel_name,
            calibration_data,
            electronics_board_name,
            temperature_coefficient_name,
        )
    elif radiance_method == RadianceMethod.PHYSICAL:
        nw_per_dn = calculate_physical_nanowatts_per_dn(
            pwm_dn_data, temperature_dn_data, calibration_data, electronics_board_name
        )
    else:
        raise ValueError("Radiance method must be either numerical or physical")

    radiance = calculate_radiance_from_dn(pwm_dn_data, nw_per_dn, channel_name, calibration_data)
    return radiance


def calculate_radiance_from_dn(
    pwm_dn_data: float | pd.Series,
    nw_per_dn: float,
    channel_name: ChannelName,
    calibration_data: LiberaGroundCalibration,
):
    """
    Calculate the radiance from the count data (dn's) in a science data packet using a given nw_per_dn conversion
    factor.

    Parameters
    ----------
    pwm_dn_data : Union[float, pd.Series]
        The pulse-width-modulation (PWM) data in a count format (dn) from the detector
    nw_per_dn : float
        The nano-watts per count (dn) conversion factor
    channel_name : ChannelName
        The channel name of the detector. Options are SHORTWAVE, TOTAL, LONGWAVE, and SPLIT_SHORTWAVE
    calibration_data : LiberaGroundCalibration
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the LiberaGroundCalibration object

    Returns
    -------
    radiance : float
        The radiance of the detector in units of W m^-2 sr^-1
    """
    channel_calibration_info = calibration_data.channels[channel_name.value]
    # Convert the pwm data as dn into a nW power using the derived conversion from the above coefficients
    power_nw = nw_per_dn * pwm_dn_data

    # Convert the power to radiance in units of W m^-2 sr^-1
    power_watts = power_nw * 1e-9
    radiance = power_watts / (channel_calibration_info.collection_area * channel_calibration_info.solid_angle)

    return radiance


def calculate_numerical_nanowatts_per_dn(
    dark_pwm_dn_data: float | pd.Series,
    temperature_dn_data: [float, pd.Series],
    channel_name: ChannelName,
    calibration_data: LiberaGroundCalibration,
    electronics_board_name: BoardName = BoardName.EMFPE,
    temperature_coefficient_name: TemperatureCoefficient = TemperatureCoefficient.BENCH_COEFFICIENTS,
) -> float:
    """
    Calculate the nano-watts per count (dn) conversion factor using the numerical method

    Parameters
    ----------
    dark_pwm_dn_data : Union[float, pd.Series]
        The dark pulse-width-modulation (PWM) data in a count format (dn) from the detector
    temperature_dn_data : Union[float, pd.Series]
        The temperature data in a count format (dn) from the detector
    channel_name : ChannelName
        The channel name of the detector. Options are SHORTWAVE, TOTAL, LONGWAVE, and SPLIT_SHORTWAVE
    calibration_data : LiberaGroundCalibration
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the LiberaGroundCalibration object
    electronics_board_name : BoardName, optional
        The name of the electronics board used for the detector. Default is EMFPE
    temperature_coefficient_name : HousekeepingTemperatureCoefficient, optional
        The name of the temperature coefficient used for the detector. Default is BENCH_COEFFICIENTS

    Returns
    -------
    nw_per_dn : float
        The nano-watts per count (dn) conversion factor
    """

    # Filter down to the relevant calibration data for easy passing through functions
    electronics_calibration_info = calibration_data.boards[electronics_board_name.value]
    channel_calibration_info = calibration_data.channels[channel_name.value]
    board_temp_coefficients = calibration_data.housekeeping_temperature_coefficients[temperature_coefficient_name.value]

    # Extract numerical coefficients used for the radiance calculation
    radiance_coefficients = channel_calibration_info.radiance_coefficients

    # Calculate a dark PWM level as a baseline
    active_dc = calculate_dark_level_duty_cycle(dark_pwm_dn_data, electronics_calibration_info)

    # Calculate the relevant temperatures needed for the radiance calculation
    bench_temp = calculate_temperature_from_dn(temperature_dn_data, board_temp_coefficients)
    reference_temp = radiance_coefficients.t0_per_dn

    nw_per_dn = (
        radiance_coefficients.constant_offset
        + radiance_coefficients.temp_difference_linear * (bench_temp - reference_temp)
        + radiance_coefficients.temp_difference_quadratic * (bench_temp - reference_temp) ** 2
        + radiance_coefficients.mean_state_offset_linear * active_dc
        + radiance_coefficients.mean_state_offset_quadratic * active_dc**2
        + radiance_coefficients.mean_state_vs_temperature_crossover * (bench_temp - reference_temp) * active_dc
    )

    return nw_per_dn


def calculate_dark_level_duty_cycle(
    pwm_dn_data: float | pd.Series, electronics_calibration_data: BoardCalibrations
) -> float:
    """
    Placeholder for calculating the dark level duty cycle as it becomes clearer on how the instrument team does this.
    Currently using the median of a full sample as each detector is turned off most of the time

    Parameters
    ----------
    pwm_dn_data : Union[float, pd.Series]
        The pulse-width-modulation (PWM) data in a count format (dn) from the detector
    electronics_calibration_data : BoardCalibrations
        The calibration data for the electronics board. This includes the maximum PWM value for the detector

    Returns
    -------
    dark_level : float
        The best estimate of the dark level duty cycle in dn units
    """
    dark_level = np.median(pwm_dn_data / electronics_calibration_data.max_pwm)
    return dark_level


def calculate_temperature_from_dn(
    temperature_dn_data: float | pd.Series, temperature_coefficients: TemperatureCoefficients
):
    """
    Calculate a temperature from the dn's in a science data packet using a cubic fit created by the instrument team

    Parameters
    ----------
    temperature_dn_data : Union[float, pd.Series]
        The temperature data in a count format (dn) from the detector
    temperature_coefficients : TemperatureCoefficients
        The temperature coefficients used to convert the dn's to a temperature

    Returns
    -------
    temperature : float
        The temperature in degrees Celsius
    """
    temperature = (
        temperature_coefficients.constant
        + temperature_coefficients.linear * temperature_dn_data
        + temperature_coefficients.quadratic * temperature_dn_data**2
        + temperature_coefficients.cubic * temperature_dn_data**3
    )

    return temperature


def calculate_physical_nanowatts_per_dn(
    active_pwm: float | pd.Series,
    heat_sink_temp: float,
    channel_name: ChannelName,
    ground_calibration_info: LiberaGroundCalibration,
    board_name: BoardName = BoardName.EMFPE,
) -> float:
    """
    Calculate the nano-watts per count (dn) conversion factor using physical principles. Extended discussion of this
    method can be found in the Libera Radiometer Algorithm Theoretical Basis Document (ATBD) and the provided learning
    notebooks in this repository.

    Parameters
    ----------
    active_pwm : Union[float, pd.Series]
        The active pulse-width-modulation (PWM) data in a count format (dn) from the detector
    heat_sink_temp : float
        The temperature of the heat sink in degrees Celsius
    channel_name : ChannelName
        The channel name of the detector. Options are SHORTWAVE, TOTAL, LONGWAVE, and SPLIT_SHORTWAVE
    ground_calibration_info : LiberaGroundCalibration
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the LiberaGroundCalibration object
    board_name : BoardName, optional
        The name of the electronics board used for the detector. Default is EMFPE

    Returns
    -------
    nw_per_dn : float
        The nano-watts per count (dn) conversion factor
    """
    # Extract the relevant calibration information
    channel_calibration = ground_calibration_info.channels[channel_name.value]
    board_calibration = ground_calibration_info.boards[board_name.value]

    # Calculate the mean duty cycle as the dark level to compare against
    mean_duty_cycle = calculate_dark_level_duty_cycle(active_pwm, board_calibration)

    # Build the power emission equation
    # Temperature range in Kelvin
    temp_range_c = np.arange(heat_sink_temp, heat_sink_temp + 200, 0.1)
    temp_range_k = constants.convert_temperature(temp_range_c, "C", "K")
    emitted_power_range = create_emitted_power_interpolation(temp_range_k, heat_sink_temp, channel_calibration)

    # Starting temperature guess in Celsius
    # These will be iteratively updated
    previous_temp = 0
    estimated_temp = 35
    iteration = 0

    # Iterate until the temperature converges to a 1 ppm level at the nW level (1e-9*1e-6)
    while abs(estimated_temp - previous_temp) > 1e-15 and iteration < 10:
        previous_temp = estimated_temp
        max_heater_power = calculate_heater_max_power(
            estimated_temp,
            heat_sink_temp,
            channel_specific_calibration_data=channel_calibration,
            board_specific_calibration_data=board_calibration,
        )

        thermistor_power = calculate_thermistor_power(
            estimated_temp,
            heat_sink_temp,
            channel_specific_calibration_data=channel_calibration,
            board_specific_calibration_data=board_calibration,
        )

        # Total Power from the circuit perspective
        mean_heater_power = max_heater_power * mean_duty_cycle
        total_power = mean_heater_power + thermistor_power

        # Estimate the temperature using the emitted power calculation from the beginning
        estimated_temp_kelvin = np.interp(total_power, emitted_power_range, temp_range_k)
        estimated_temp = constants.convert_temperature(estimated_temp_kelvin, "K", "C")

        logger.debug(f"Iteration: {iteration}. Estimated Temperature:  {estimated_temp}")
        iteration += 1

    if iteration == 10:
        logger.error("Temperature calculation did not converge")
        raise ValueError("Temperature calculation did not converge")

    watts_per_dn = max_heater_power / board_calibration.max_pwm

    return watts_per_dn * 1e9


def calculate_resistance_from_temp(
    temperature_data: float | pd.Series,
    channel_specific_calibration_information: ChannelCalibrations,
    trace_path: DetectorTracePath,
    detector_choice: DetectorType = DetectorType.ACTIVE,
) -> float:
    """
    Calculate the resistance of a detector from temperature data using ground calibration information derived by the
    instrument team.

    Parameters
    ----------
    temperature_data : Union[float, pd.Series]
        The temperature data in degrees Celsius from the detector
    channel_specific_calibration_information : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    trace_path : DetectorTracePath
        The path of the detector trace. Options are HEATER and THERMISTOR
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE

    Returns
    -------
    resistance : float
        The resistance of the detector trace in Ohms
    """
    if detector_choice == DetectorType.ACTIVE:
        detector_specific_info = channel_specific_calibration_information.active_detector
    elif detector_choice == DetectorType.REFERENCE:
        detector_specific_info = channel_specific_calibration_information.reference_detector
    else:
        raise ValueError("Detector choice must be either active or reference")

    if trace_path == DetectorTracePath.HEATER:
        trace_calibrations = detector_specific_info.heater
    elif trace_path == DetectorTracePath.THERMISTOR:
        trace_calibrations = detector_specific_info.thermistor
    else:
        raise ValueError("Trace path must be either heater or thermistor")

    # Extract the linear slope and offset from the calibration data
    linear_slope = trace_calibrations.t_vs_r_slope
    linear_offset = trace_calibrations.t_vs_r_intercept
    base_temperature = channel_specific_calibration_information.t0

    # Calculate resistance from temperature data
    return linear_slope * (temperature_data - base_temperature) + linear_offset


class GeometricResistance:
    """Class to hold the geometric resistances of a detector trace, the silicone resistance, the leg resistance, and
    the detector resistance.
    """

    def __init__(self, *, silicone_resistance: float, leg_resistance: float, detector_resistance: float):
        self.silicone_resistance = silicone_resistance
        self.leg_resistance = leg_resistance
        self.detector_resistance = detector_resistance


def calculate_geometric_resistance(
    trace_resistance: float | pd.Series,
    heat_sink_resistance: float,
    channel_specific_calibration_data: ChannelCalibrations,
    trace_path: DetectorTracePath,
    detector_choice: DetectorType = DetectorType.ACTIVE,
) -> GeometricResistance:
    """
    Calculate the geometric resistance of a detector trace using ground calibration information

    Parameters
    ----------
    trace_resistance : Union[float, pd.Series]
        The resistance of the detector trace in Ohms
    heat_sink_resistance : float
        The resistance of the heat sink in Ohms
    channel_specific_calibration_data : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    trace_path : DetectorTracePath
        The path of the detector trace. Options are HEATER and THERMISTOR
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE

    Returns
    -------
    total_resistance : GeometricResistance
        The geometric resistances of the detector trace
    """
    if detector_choice == DetectorType.ACTIVE:
        detector_specific_info = channel_specific_calibration_data.active_detector
    elif detector_choice == DetectorType.REFERENCE:
        detector_specific_info = channel_specific_calibration_data.reference_detector
    else:
        raise ValueError("Detector choice must be either active or reference")

    if trace_path == DetectorTracePath.HEATER:
        circuit_path_specific_cal_data = detector_specific_info.heater
    elif trace_path == DetectorTracePath.THERMISTOR:
        circuit_path_specific_cal_data = detector_specific_info.thermistor
    else:
        raise ValueError("Trace path must be either heater or thermistor")

    # This is the resistance of the silicone the detector is attached to
    silicone = circuit_path_specific_cal_data.silicone_trace_fraction * heat_sink_resistance
    # Assume the resistance across the legs is half from the silicone and half from the detector trace
    legs = 0.5 * circuit_path_specific_cal_data.leg_trace_fraction * (trace_resistance + heat_sink_resistance)
    # Use the trace resistance for the meander across the detector
    detector = circuit_path_specific_cal_data.meander_trace_fraction * trace_resistance

    total_resistance = GeometricResistance(
        silicone_resistance=silicone, leg_resistance=legs, detector_resistance=detector
    )
    return total_resistance


def calculate_heater_current(
    geometric_resistances: GeometricResistance,
    channel_specific_calibration_data: ChannelCalibrations,
    board_specific_calibration_data: BoardCalibrations,
    detector_choice: DetectorType = DetectorType.ACTIVE,
) -> float:
    """
    Calculate the current across the heater trace of a detector

    Parameters
    ----------
    geometric_resistances : GeometricResistance
        The geometric resistances of the detector trace
    channel_specific_calibration_data : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    board_specific_calibration_data : BoardCalibrations
        The calibration data for the electronics board. This includes the maximum PWM value for the detector
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE

    Returns
    -------
    current : float
        The current across the heater trace in Amps
    """
    if detector_choice == DetectorType.ACTIVE:
        detector_specific_cal_data = channel_specific_calibration_data.active_detector
    elif detector_choice == DetectorType.REFERENCE:
        detector_specific_cal_data = channel_specific_calibration_data.reference_detector
    else:
        raise ValueError("Detector choice must be either active or reference")

    top_resistance = detector_specific_cal_data.top_resistance
    # Calculate the total resistance across the heater
    total_resistance = top_resistance + (
        geometric_resistances.detector_resistance
        + geometric_resistances.leg_resistance
        + geometric_resistances.silicone_resistance
    )

    # Calculate the total voltage on that path
    total_voltage = board_specific_calibration_data.voltage_reference  # Older + detector_specific_cal_data.vos

    # Calculate the current across the heater
    current = total_voltage / total_resistance
    return current


def calculate_heater_max_power(
    estimated_temp: float | pd.Series,
    heat_sink_temp: float,
    channel_specific_calibration_data: ChannelCalibrations,
    board_specific_calibration_data: BoardCalibrations,
    detector_choice: DetectorType = DetectorType.ACTIVE,
):
    """
    Compute the mean power along the heating circuit path

    Parameters
    ----------
    estimated_temp : Union[float, pd.Series]
        The estimated temperature of the detector in degrees Celsius
    heat_sink_temp : float
        The temperature of the heat sink in degrees Celsius
    channel_specific_calibration_data : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    board_specific_calibration_data : BoardCalibrations
        The calibration data for the electronics board. This includes the maximum PWM value for the detector
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE

    Returns
    -------
    max_heater_power : float
        The maximum power along the heating circuit path in Watts
    """

    # Calculate the heater resistance from temperature
    heater_resistance = calculate_resistance_from_temp(
        estimated_temp,
        channel_specific_calibration_data,
        detector_choice=detector_choice,
        trace_path=DetectorTracePath.HEATER,
    )
    # Calculate the heat_sink resistance from temperature
    heater_heat_sink_resistance = calculate_resistance_from_temp(
        heat_sink_temp,
        channel_specific_calibration_data,
        detector_choice=detector_choice,
        trace_path=DetectorTracePath.HEATER,
    )
    # Calculate the resistance using the geometry of the detector
    heater_geometric_resistance = calculate_geometric_resistance(
        heater_resistance,
        heater_heat_sink_resistance,
        channel_specific_calibration_data,
        trace_path=DetectorTracePath.HEATER,
        detector_choice=detector_choice,
    )
    # Calculate the relevant resistances using the geometry of the detector
    heater_effective_resistance = heater_geometric_resistance.detector_resistance + (
        heater_geometric_resistance.leg_resistance / 2
    )

    # Calculate the current across the heater
    heater_current = calculate_heater_current(
        heater_geometric_resistance,
        channel_specific_calibration_data,
        board_specific_calibration_data,
        detector_choice=detector_choice,
    )
    # Calculate the power over the heater
    # P = I^2 * R
    max_heater_power = np.power(heater_current, 2) * heater_effective_resistance

    return max_heater_power


def calculate_thermistor_power(
    estimated_temp: float | pd.Series,
    heat_sink_temp: float,
    channel_specific_calibration_data: ChannelCalibrations,
    board_specific_calibration_data: BoardCalibrations,
    detector_choice: DetectorType = DetectorType.ACTIVE,
) -> float:
    """
    Compute the power along the thermistor circuit path.

    Parameters
    ----------
    estimated_temp : Union[float, pd.Series]
        The estimated temperature of the detector in degrees Celsius
    heat_sink_temp : float
        The temperature of the heat sink in degrees Celsius
    channel_specific_calibration_data : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    board_specific_calibration_data : BoardCalibrations
        The calibration data for the electronics board. This includes the maximum PWM value for the detector
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE
    """
    # Calculate the thermistor resistance from temperature
    thermistor_resistance = calculate_resistance_from_temp(
        estimated_temp,
        channel_specific_calibration_data,
        detector_choice=detector_choice,
        trace_path=DetectorTracePath.THERMISTOR,
    )
    # Calculate the heat_sink resistance from temperature
    thermistor_heat_sink_resistance = calculate_resistance_from_temp(
        heat_sink_temp,
        channel_specific_calibration_data,
        detector_choice=detector_choice,
        trace_path=DetectorTracePath.THERMISTOR,
    )
    # Calculate the power over the thermistor
    geometry_resistances = calculate_geometric_resistance(
        thermistor_resistance,
        thermistor_heat_sink_resistance,
        channel_specific_calibration_data,
        trace_path=DetectorTracePath.THERMISTOR,
    )
    # Calculate the relevant resistances using the geometry of the detector
    thermistor_lead_resistance = geometry_resistances.silicone_resistance + geometry_resistances.leg_resistance / 2
    thermistor_effective_resistance = geometry_resistances.detector_resistance + geometry_resistances.leg_resistance / 2

    if detector_choice == DetectorType.ACTIVE:
        detector_specific_info = channel_specific_calibration_data.active_detector
    elif detector_choice == DetectorType.REFERENCE:
        detector_specific_info = channel_specific_calibration_data.reference_detector
    else:
        raise ValueError("Detector choice must be either active or reference")

    bridge_resistance = detector_specific_info.bridge_resistance
    sine_rms = board_specific_calibration_data.sine_rms

    # Calculate the current across the thermistor
    thermistor_rms_current = (2 * sine_rms) / (
        thermistor_lead_resistance + thermistor_effective_resistance + bridge_resistance
    )
    # Calculate the power over the thermistor
    # P = I^2 * R
    thermistor_power = np.power(thermistor_rms_current, 2) * thermistor_effective_resistance

    return thermistor_power


def create_emitted_power_interpolation(
    temperature_range: float | pd.Series,
    heat_sink_temp: float,
    channel_specific_calibration_data: ChannelCalibrations,
    detector_choice: DetectorType = DetectorType.ACTIVE,
) -> np.ndarray:
    """
    Create an interpolation of the emitted power based on a temperature range and heat transfer principles

    Parameters
    ----------
    temperature_range : Union[float, pd.Series]
        The temperature range in Kelvin
    heat_sink_temp : float
        The temperature of the heat sink in degrees Celsius
    channel_specific_calibration_data : ChannelCalibrations
        The calibration data for the detector. This includes the radiance coefficients, temperature coefficients, and
        other relevant calibration information stored in the ChannelCalibrations object
    detector_choice : DetectorType, optional
        The type of detector. Options are ACTIVE and REFERENCE. Default is ACTIVE

    Returns
    -------
    power_emitted_range_W : np.ndarray
        The emitted power in the temperature range in Watts
    """
    if detector_choice == DetectorType.ACTIVE:
        active_detector_calibration_info = channel_specific_calibration_data.active_detector
    elif detector_choice == DetectorType.REFERENCE:
        active_detector_calibration_info = channel_specific_calibration_data.reference_detector
    else:
        raise ValueError("Detector choice must be either active or reference")

    t_base = constants.convert_temperature(heat_sink_temp, "C", "K")
    # Conductive Power
    cond_power = active_detector_calibration_info.conductive_constant * (temperature_range - t_base)
    # Radiated Power
    rad_power = (
        constants.Stefan_Boltzmann
        * active_detector_calibration_info.radiative_constant
        * (np.power(temperature_range, 4) - np.power(t_base, 4))
    )
    # Total Emitted Power = Conductive Power + Radiated Power (in uW, microWatts)
    power_emitted_range_uW = cond_power + rad_power

    # Convert to Watts
    power_emitted_range_W = power_emitted_range_uW / 1e6

    return power_emitted_range_W


def calibrate_and_downsample_radiometer_data(rad_data: xr.Dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Apply gain calibration and downsample radiometer data.

    For each radiometer channel, this function:
    1. Extracts the raw digital number (DN) measurements
    2. Applies gain calibration using ground calibration response functions
    3. Downsamples the calibrated data to 100Hz

    Parameters
    ----------
    rad_data : xr.Dataset
        Radiometer sample dataset containing raw measurements and timestamps.

    Returns
    -------
    np.ndarray
        Timestamps downsampled to 100Hz.
    dict[str, np.ndarray]
        Dictionary mapping channel names (e.g., 'sw', 'lw', 'total', 'ssw') to
        calibrated and downsampled radiometer data arrays.

    Warnings
    --------
    Logs a warning if no variable is found matching a channel's enum identifier.
    """
    raw_times = rad_data["RAD_SAMPLE_FPE_TIME"].values

    calibrated_data_by_channel = {}
    calibrated_data_points = 0

    for channel_name, channel_properties in CALIBRATION_DATA.channels.items():
        channel_variable = find_channel_variable(rad_data, channel_properties.channel_enum)

        if channel_variable is None:
            logger.warning(f"No variable found for channel {channel_name}")
            continue

        channel_dns = rad_data[channel_variable].to_numpy()

        # Apply gain calibration
        channel_calibrated_rad_data = apply_gain_calibration(
            channel_dns,
            get_ground_cal_response_function(freqs=np.arange(0, len(channel_dns) / 2 + 1)),
            len(channel_dns),
        )

        # Downsample to 100Hz
        calibrated_100hz_rad_data = downsample_libera_signal(
            channel_calibrated_rad_data, from_rate=_RAD_SAMPLE_HZ, to_rate=_L1B_OUTPUT_HZ
        )
        calibrated_data_by_channel[channel_name] = calibrated_100hz_rad_data
        if not calibrated_data_points:
            calibrated_data_points = len(calibrated_100hz_rad_data)

    # TODO[LIBSDC-720]: double check with Dave on timestamp downsampling
    if not np.issubdtype(raw_times.dtype, np.datetime64):
        raise ValueError("RAD_SAMPLE_FPE_TIME must be datetime64[ns]; open L1A NetCDF inputs with decode_times=True.")
    factor = decimation_factor(_RAD_SAMPLE_HZ, _L1B_OUTPUT_HZ)
    timestamps = raw_times[::factor][:calibrated_data_points]
    return timestamps, calibrated_data_by_channel


def interpolate_temperatures(timestamps: np.ndarray, nom_hk_data: xr.Dataset) -> pd.Series:
    """
    Interpolate temperature data to radiometer sampling frequency.

    Housekeeping temperature measurements are sampled at a lower frequency than radiometer data (1Hz vs. 100Hz after
    downsampling). This function interpolates temperatures to match the radiometer timestamps.

    Parameters
    ----------
    timestamps : np.ndarray
        Target timestamps at radiometer sampling frequency (100Hz).
    nom_hk_data : xr.Dataset
        Nominal housekeeping dataset containing temperature measurements.

    Returns
    -------
    pd.Series
        Temperature values interpolated to match radiometer timestamps.

    Notes
    -----
    Linear interpolation is used, with 'PACKET_ICIE_TIME' and 'ICIE__FPE_TSCOPE_TEMP' from housekeeping data.
    """
    # TODO[LIBSDC-713]: Compare interpolation of temperature data with average temperature for period
    #     and consult IE team with results.
    hk_times = nom_hk_data["PACKET_ICIE_TIME"].values
    if not np.issubdtype(hk_times.dtype, np.datetime64):
        raise ValueError("PACKET_ICIE_TIME must be datetime64[ns]; open L1A NetCDF inputs with decode_times=True.")
    hk_x = hk_times.astype(np.int64)
    ts_x = np.asarray(timestamps).astype(np.int64)

    return pd.Series(
        np.interp(
            ts_x,
            hk_x,
            nom_hk_data["ICIE__FPE_TSCOPE_TEMP"].to_series().to_numpy(dtype=np.float64),
        )
    )


def calculate_radiances(
    calibrated_data_by_channel: dict[str, np.ndarray], interpolated_temperatures: pd.Series
) -> dict[str, np.ndarray]:
    """
    Calculate radiance from calibrated and downsampled dns.

    Parameters
    ----------
    calibrated_data_by_channel : dict[str, np.ndarray]
        Dictionary of calibrated radiometer data by channel name.
    interpolated_temperatures : pd.Series
        Temperature measurements at radiometer sampling frequency.
    calibration_data : LiberaGroundCalibration
        Ground calibration parameters containing conversion coefficients.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping channel names to calculated radiance arrays in W/m²/sr.

    Warnings
    --------
    Logs a warning if a channel string cannot be converted to a ChannelName enum.
    """
    calculated_radiance_by_channel = {}

    for channel, dataset in calibrated_data_by_channel.items():
        channel_enum = get_channel_name_enum(channel)
        if channel_enum is None:
            logger.warning(f"Could not convert channel string '{channel}' to enum")
            continue

        calculated_radiance = calculate_radiance(
            pd.Series(dataset),
            interpolated_temperatures,
            channel_name=channel_enum,
            calibration_data=CALIBRATION_DATA,
        )
        if isinstance(calculated_radiance, np.ndarray):
            calculated_radiance_by_channel[channel] = calculated_radiance
        else:
            calculated_radiance_by_channel[channel] = calculated_radiance.to_numpy()

    return calculated_radiance_by_channel
