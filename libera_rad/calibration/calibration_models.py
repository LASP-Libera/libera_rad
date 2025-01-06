""" Pydantic models for the calibration data to simplify their usage in the libera_rad package"""
# Installed
from pydantic import BaseModel


class BoardCalibrations(BaseModel):
    """ Pydantic model for the calibration data for the PCBs used with the detectors """
    name: str
    voltage_reference: float
    vrms_per_vamp: float
    max_pwm: int
    sine_rms: float


class TraceCalibrations(BaseModel):
    """ Pydantic model for the calibration data for the detector traces (heater vs thermistor) for a given detector """
    silicone_trace_fraction: float
    leg_trace_fraction: float
    meander_trace_fraction: float
    t_vs_r_intercept: float
    t_vs_r_slope: float


class DetectorCalibrations(BaseModel):
    """ Pydantic model for the calibration data for the detectors """
    heater: TraceCalibrations
    thermistor: TraceCalibrations
    top_resistance: float
    bridge_resistance: float
    vos: float
    conductive_constant: float
    radiative_constant: float


class RadianceCoefficients(BaseModel):
    """ Pydantic model for the radiance coefficients for the detectors for the numeric calculations """
    t0_per_dn: float
    constant_offset: float
    temp_difference_linear: float
    temp_difference_quadratic: float
    mean_state_offset_linear: float
    mean_state_offset_quadratic: float
    mean_state_vs_temperature_crossover: float


class ChannelCalibrations(BaseModel):
    """ Pydantic model for the calibration data for the channels used with the detectors """
    name: str
    channel_enum: int
    module_sn: int
    carrier_sn: int
    chip_sn: str
    detector_pcb: str
    collection_area: float
    collection_area_sd: float
    solid_angle: float
    solid_angle_sd: float
    substrate_temp_offset: int
    t0: int
    radiance_coefficients: RadianceCoefficients
    active_detector: DetectorCalibrations
    reference_detector: DetectorCalibrations


class TemperatureCoefficients(BaseModel):
    """ Pydantic model for the temperature coefficients for the housekeeping information to convert from dn to
    temperature in Celsius """
    constant: float
    linear: float
    quadratic: float
    cubic: float


class LiberaGroundCalibration(BaseModel):
    """ Pydantic model for all the calibration data from the Libera ground calibration for radiance calculations """
    boards: dict[str, BoardCalibrations]
    channels: dict[str, ChannelCalibrations]
    housekeeping_temperature_coefficients: dict[str, TemperatureCoefficients]
    calibration_source: str
    calibration_version: str
    calibration_notes: str
    calibration_author: str
    calibration_author_email: str
