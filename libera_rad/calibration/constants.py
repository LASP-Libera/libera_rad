"""Constants for the libera_rad package used for retrieving calibration data"""

# Standard
from enum import Enum

import xarray as xr


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


def find_channel_variable(rad_data: xr.Dataset, channel_enum: ChannelName) -> str | None:
    """
    Find the variable name in radiometer data corresponding to a channel enum.
    This function maps the last character of a variable in rad_data:
    0 = sw
    1 = lw
    2 = total
    3 = ssw

    Parameters
    ----------
    rad_data : xr.Dataset
        Radiometer sample dataset containing channel variables.
    channel_enum : ChannelName
        Channel enum identifier to search for.

    Returns
    -------
    str or None
        Variable name matching the channel enum, or None if not found.

    Notes
    -----
    Variable names are expected to have the channel identifier as the last character, e.g., 'ICIE__' for channel enum
    '1'.
    """
    for variable in rad_data.variables:
        if str(channel_enum) == str(variable)[-1]:
            return str(variable)
    return None


def get_channel_name_enum(channel_str: str) -> ChannelName | None:
    """
    Convert channel string to ChannelName enum.

    Maps a channel name string to its corresponding ChannelName enumeration value.
    'sw' = ChannelEnum.SHORTWAVE
    'lw = ChannelEnum.LONGWAVE
    'total' = ChannelEnum.TOTAL
    'ssw' = ChannelEnum.SPLIT_SHORTWAVE

    Parameters
    ----------
    channel_str : str
        Channel name as string (e.g., 'sw', 'lw', 'total', 'ssw').

    Returns
    -------
    ChannelName or None
        Corresponding ChannelName enum member, or None if no match is found.
    """
    for channel_name in ChannelName:
        if channel_str == channel_name.value:
            return channel_name
    return None
