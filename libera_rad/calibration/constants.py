"""Constants for the libera_rad package used for retrieving calibration data"""

# Standard
from enum import Enum

import xarray as xr
from libera_utils.constants import DataProductIdentifier

COMBINER_CCSDS_KEEP_FIELDS = ["PACKET", "PACKET_ICIE_TIME", "SRC_SEQ_CTR", "PKT_LEN", "PKT_APID"]

COMBINER_CCSDS_DROP_FIELDS = [
    "VERSION",
    "TYPE",
    "SEC_HDR_FLAG",
    "SEQ_FLGS",
    "REUSABLE_SPARE_2",  # NOM-HK + PEV-SW-STAT
    "REUSABLE_SPARE_4",  # PEV-SW-STAT + RAD-SAMPLE
    "REUSABLE_SPARE_8",  # NOM-HK + PEV-SW-STAT + RAD-SAMPLE
    "REUSABLE_SPARE_16",  # PEV-SW-STAT + RAD-SAMPLE
]

#: Mapping of gain-cal OBSID → combined calibration product identifier.
COMBINER_GAIN_OBSID_TO_PRODUCT_IDENTIFIER: dict[int, DataProductIdentifier] = {
    512: DataProductIdentifier.cal_gain_combined,
}

# TODO LIBSDC-564: Update with change to OBSID numbering in
#: Mapping of shortwave-cal OBSID → combined calibration product identifier.
COMBINER_SW_OBSID_TO_PRODUCT_IDENTIFIER: dict[int, DataProductIdentifier] = {
    256: DataProductIdentifier.cal_sw_combined,  # 365 nm LED
    257: DataProductIdentifier.cal_sw_combined,  # 405 nm LED
    258: DataProductIdentifier.cal_sw_combined,  # 520 nm LED
    259: DataProductIdentifier.cal_sw_combined,  # 635 nm LED
    260: DataProductIdentifier.cal_sw_combined,  # 840 nm LED
    261: DataProductIdentifier.cal_sw_combined,  # 1550 nm LED
}

#: Mapping of longwave-cal OBSID → combined calibration product identifier.
COMBINER_LW_OBSID_TO_PRODUCT_IDENTIFIER: dict[int, DataProductIdentifier] = {
    320: DataProductIdentifier.cal_lw_temp1_combined,  #
    321: DataProductIdentifier.cal_lw_temp2_combined,
    322: DataProductIdentifier.cal_lw_temp3_combined,
}

# TODO LIBSDC-564: Update with change to OBSID numbering in
#: Mapping of solar-cal OBSID → face-level combined calibration product identifier.
#: OBSIDs per ICD:
#:   384-387: Face 1 (primary diffuser)
#:   388-391: Face 2 (secondary diffuser)
#:   392-395: Face 3 (tertiary diffuser)
COMBINER_SOLAR_OBSID_TO_PRODUCT_IDENTIFIER: dict[int, DataProductIdentifier] = {
    # Face 1
    384: DataProductIdentifier.cal_solar_face1_combined,  # Primary Face SSW Channel
    385: DataProductIdentifier.cal_solar_face1_combined,  # Primary Face TOT Channel
    386: DataProductIdentifier.cal_solar_face1_combined,  # Primary Face LW Channel
    387: DataProductIdentifier.cal_solar_face1_combined,  # Primary Face SW Channel
    # Face 2
    388: DataProductIdentifier.cal_solar_face2_combined,  # Secondary Face SSW Channel
    389: DataProductIdentifier.cal_solar_face2_combined,  # Secondary Face TOT Channel
    390: DataProductIdentifier.cal_solar_face2_combined,  # Secondary Face LW Channel
    391: DataProductIdentifier.cal_solar_face2_combined,  # Secondary Face SW Channel
    # Face 3
    392: DataProductIdentifier.cal_solar_face3_combined,  # Tertiary Face SSW Channel
    393: DataProductIdentifier.cal_solar_face3_combined,  # Tertiary Face TOT Channel
    394: DataProductIdentifier.cal_solar_face3_combined,  # Tertiary Face LW Channel
    395: DataProductIdentifier.cal_solar_face3_combined,  # Tertiary Face SW Channel
}

#: Face number (1, 2, 3) keyed by solar-cal combined product identifier.
COMBINER_SOLAR_FACE_IDENTIFIER_TO_FACE_NUM: dict[DataProductIdentifier, int] = {
    DataProductIdentifier.cal_solar_face1_combined: 1,
    DataProductIdentifier.cal_solar_face2_combined: 2,
    DataProductIdentifier.cal_solar_face3_combined: 3,
}

#: First OBSID for each solar-cal face number (used to derive ``event_pass_index``).
COMBINER_SOLAR_FACE_BASE_OBSIDS: dict[int, int] = {1: 384, 2: 388, 3: 392}


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
    1 = total
    2 = lw
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
