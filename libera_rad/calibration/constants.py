""" Constants for the libera_rad package used for retrieving calibration data """
# Standard
from enum import Enum


class BoardName(Enum):
    """Enum of valid board names"""
    P01 = "p01"
    P02 = "p02"
    P03 = "p03"
    PTEST = "pTest"
    EMFPE = "emfpe"


class ChannelName(Enum):
    """Enum of valid detector names"""
    TOTAL = "total"
    SHORTWAVE = "sw"
    LONGWAVE = "lw"
    SPLIT_SHORTWAVE = "ssw"


class DetectorTracePath(Enum):
    """Enum of valid detector circuit paths"""
    HEATER = "heater"
    THERMISTOR = "thermistor"


class DetectorType(Enum):
    """Enum of valid detector types"""
    ACTIVE = "active"
    REFERENCE = "reference"


class HousekeepingTemperatureCoefficient(Enum):
    """Enum of valid housekeeping temperature fields"""
    BENCH_COEFFICIENTS = "bench_coefficients"
    LT1000_COEFFICIENTS = "lt1000_coefficients"
    CAES_COEFFICIENTS = "caes_coefficients"
    HEATER_ADC_COEFFICIENTS = "heater_adc_coefficients"


class RadianceMethod(Enum):
    """Enum of valid radiance methods"""
    NUMERICAL = "numerical"
    PHYSICAL = "physical"
