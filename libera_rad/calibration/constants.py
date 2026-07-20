"""Constants for the libera_rad package used for retrieving calibration data"""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import NomHkObsidSource, ObsIdKind, get_obsid_spec, iter_trim_eligible

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

#: Environment variable that selects the calibration ObsID for ``cal-combine``.
LIBERA_CAL_OBSID_ENV = "LIBERA_CAL_OBSID"

CalFamily = Literal["gain", "swc", "lwc", "solar"]


@dataclass(frozen=True)
class CalEventSpec:
    """Specification for one ObsID-specific calibration combine event."""

    obsid: int
    cal_product: DataProductIdentifier
    trimmed_product: DataProductIdentifier
    family: CalFamily
    companion_products: tuple[DataProductIdentifier, ...]
    time_variable: str


_GAIN_COMPANIONS = (
    DataProductIdentifier.l1a_icie_rad_full_decoded,
    DataProductIdentifier.l1a_icie_cal_full_decoded,
)

_SWC_COMPANIONS = (
    DataProductIdentifier.l1a_icie_cal_sample_decoded,
    DataProductIdentifier.l1a_icie_rad_sample_decoded,
    DataProductIdentifier.l1a_pec_sw_stat_decoded,
    DataProductIdentifier.l1a_pev_sw_stat_decoded,
)

_LWC_COMPANIONS = (
    DataProductIdentifier.l1a_icie_rad_sample_decoded,
    DataProductIdentifier.l1a_pec_sw_stat_decoded,
    DataProductIdentifier.l1a_pev_sw_stat_decoded,
)

_SOLAR_COMPANIONS = (
    DataProductIdentifier.l1a_icie_rad_sample_decoded,
    DataProductIdentifier.l1a_pev_sw_stat_decoded,
)


def _rad_products(obsid: int) -> tuple[DataProductIdentifier, DataProductIdentifier]:
    """Resolve CAL and TRIMMED ProductIDs from the shared libera_utils ObsID registry."""
    spec = get_obsid_spec(NomHkObsidSource.RAD, obsid)
    if spec.cal_product is None or spec.trimmed_product is None:
        raise ValueError(f"RAD ObsID {obsid} is not a trim-eligible calibration event")
    return spec.cal_product, spec.trimmed_product


def _gain(obsid: int) -> CalEventSpec:
    cal, trimmed = _rad_products(obsid)
    return CalEventSpec(
        obsid=obsid,
        cal_product=cal,
        trimmed_product=trimmed,
        family="gain",
        companion_products=_GAIN_COMPANIONS,
        time_variable="RAD_FULL_PACKET_ICIE_TIME",
    )


def _swc(obsid: int) -> CalEventSpec:
    cal, trimmed = _rad_products(obsid)
    return CalEventSpec(
        obsid=obsid,
        cal_product=cal,
        trimmed_product=trimmed,
        family="swc",
        companion_products=_SWC_COMPANIONS,
        time_variable="RAD_SAMPLE_PACKET_ICIE_TIME",
    )


def _lwc(obsid: int) -> CalEventSpec:
    cal, trimmed = _rad_products(obsid)
    return CalEventSpec(
        obsid=obsid,
        cal_product=cal,
        trimmed_product=trimmed,
        family="lwc",
        companion_products=_LWC_COMPANIONS,
        time_variable="RAD_SAMPLE_PACKET_ICIE_TIME",
    )


def _solar(obsid: int) -> CalEventSpec:
    cal, trimmed = _rad_products(obsid)
    return CalEventSpec(
        obsid=obsid,
        cal_product=cal,
        trimmed_product=trimmed,
        family="solar",
        companion_products=_SOLAR_COMPANIONS,
        time_variable="NOM_HK_PACKET_ICIE_TIME",
    )


#: Mapping of radiometer calibration ObsID → event specification.
#: Product IDs come from ``libera_utils.obsids``; family/companions stay local.
CAL_EVENT_BY_OBSID: dict[int, CalEventSpec] = {
    # Gain
    512: _gain(512),
    # Shortwave LED
    256: _swc(256),
    257: _swc(257),
    258: _swc(258),
    259: _swc(259),
    260: _swc(260),
    261: _swc(261),
    # Longwave blackbody
    320: _lwc(320),
    321: _lwc(321),
    322: _lwc(322),
    # Solar — Face 1 (primary)
    384: _solar(384),
    385: _solar(385),
    386: _solar(386),
    387: _solar(387),
    # Solar — Face 2 (secondary)
    388: _solar(388),
    389: _solar(389),
    390: _solar(390),
    391: _solar(391),
    # Solar — Face 3 (tertiary)
    392: _solar(392),
    393: _solar(393),
    394: _solar(394),
    395: _solar(395),
}

# Sanity: every CAL_EVENT_BY_OBSID key must exist as RAD_CAL in libera_utils.
# Utils may list additional RAD_CAL ObsIDs (e.g. lunar) before rad combiners exist.
_RAD_CAL_OBSIDS = {s.obsid for s in iter_trim_eligible(NomHkObsidSource.RAD) if s.kind is ObsIdKind.RAD_CAL}
_EXTRA = set(CAL_EVENT_BY_OBSID) - _RAD_CAL_OBSIDS
if _EXTRA:
    raise RuntimeError(
        f"CAL_EVENT_BY_OBSID contains ObsIDs missing from libera_utils RAD_CAL registry: {sorted(_EXTRA)}"
    )

#: Face number (1, 2, 3) for solar-cal ObsIDs (used for product attributes).
SOLAR_OBSID_TO_FACE_NUM: dict[int, int] = {
    **dict.fromkeys(range(384, 388), 1),
    **dict.fromkeys(range(388, 392), 2),
    **dict.fromkeys(range(392, 396), 3),
}

#: First OBSID for each solar-cal face (used to derive ``event_pass_index``).
SOLAR_FACE_BASE_OBSIDS: dict[int, int] = {1: 384, 2: 388, 3: 392}


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
