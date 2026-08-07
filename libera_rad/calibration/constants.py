"""Constants for libera_rad calibration and L1B radiance processing.

ObsID → CAL/TRIMMED ProductIDs live in ``libera_utils.obsids``. This module
owns rad-only merge recipes (family companions / time variables), combiner
CCSDS field lists, and L1B science enums.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import NomHkObsidSource, ObsIdKind, ObsIdSpec, get_obsid_spec, iter_trim_eligible

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

#: Time coordinate every cal product is named from.
#:
#: NOM-HK defines the calibration event: ``select_and_slice_event_inputs`` derives the window
#: from NOM-HK packet time and every companion is trimmed to it, so NOM-HK is the only time
#: base shared by all families. Naming from a companion instead describes one stream rather
#: than the event — the gain product previously named from ``RAD_FULL_PACKET_ICIE_TIME`` and
#: stamped a range four minutes shorter than the CAL data it contained. This is deliberately a
#: single constant rather than a per-family setting so no family can reintroduce that.
#:
#: The window is not a strict envelope of the merged product. Companions are trimmed a whole
#: packet at a time, so an edge packet extends past the window by up to its own sample span,
#: and an FPE-skewed packet can extend further (a RAD-SAMPLE packet stamped 16 s past the
#: window while its samples land inside it has been observed in ground-test data).
CAL_PRODUCT_TIME_VARIABLE = "NOM_HK_PACKET_ICIE_TIME"

CalFamily = Literal["gain", "swc", "lwc", "solar"]


@dataclass(frozen=True)
class CalEventSpec:
    """Specification for one ObsID-specific calibration combine event."""

    obsid: int
    cal_product: DataProductIdentifier
    trimmed_product: DataProductIdentifier
    family: CalFamily
    companion_products: tuple[DataProductIdentifier, ...]
    #: Time coordinate the product filename is stamped from; always
    #: :data:`CAL_PRODUCT_TIME_VARIABLE`.
    time_variable: str


@dataclass(frozen=True)
class _FamilyConfig:
    """Rad-algorithm merge recipe for one calibration family."""

    companion_products: tuple[DataProductIdentifier, ...]


#: Family → companion products. ProductIDs come from libera_utils. All families are named
#: from :data:`CAL_PRODUCT_TIME_VARIABLE`.
_FAMILY_CONFIGS: dict[CalFamily, _FamilyConfig] = {
    "gain": _FamilyConfig(
        companion_products=(
            DataProductIdentifier.l1a_icie_rad_full_decoded,
            DataProductIdentifier.l1a_icie_cal_full_decoded,
        ),
    ),
    "swc": _FamilyConfig(
        companion_products=(
            DataProductIdentifier.l1a_icie_cal_sample_decoded,
            DataProductIdentifier.l1a_icie_rad_sample_decoded,
            DataProductIdentifier.l1a_pec_sw_stat_decoded,
            DataProductIdentifier.l1a_pev_sw_stat_decoded,
        ),
    ),
    "lwc": _FamilyConfig(
        companion_products=(
            DataProductIdentifier.l1a_icie_rad_sample_decoded,
            DataProductIdentifier.l1a_pec_sw_stat_decoded,
            DataProductIdentifier.l1a_pev_sw_stat_decoded,
        ),
    ),
    "solar": _FamilyConfig(
        companion_products=(
            DataProductIdentifier.l1a_icie_rad_sample_decoded,
            DataProductIdentifier.l1a_pev_sw_stat_decoded,
        ),
    ),
    # TODO [LIBSDC-811]: Add lunar cals
}


def family_from_cal_product(cal_product: DataProductIdentifier) -> CalFamily | None:
    """Map a CAL ProductID to a rad cal-combine family, or ``None`` if unsupported.

    Supported prefixes match implemented families (``GAIN``, ``NOISE``, ``SWC-``,
    ``LWC-``, ``SOLAR-``). Lunar and other RAD_CAL products return ``None`` until a
    family config is added.

    ``NOISE`` (ObsID 515) maps to the ``gain`` family: it is a distinct calibration
    event and CAL product, but it was split out of the original combined gain/noise
    ObsID 512 and is combined with the same full-rate merge recipe.
    """
    value = cal_product.value
    if value in ("GAIN", "NOISE"):
        return "gain"
    if value.startswith("SWC-"):
        return "swc"
    if value.startswith("LWC-"):
        return "lwc"
    if value.startswith("SOLAR-"):
        return "solar"
    return None


def _cal_event_from_obsid_spec(obsid_spec: ObsIdSpec) -> CalEventSpec | None:
    """Build a ``CalEventSpec`` when the utils entry maps to a supported family."""
    if obsid_spec.cal_product is None or obsid_spec.trimmed_product is None:
        return None
    family = family_from_cal_product(obsid_spec.cal_product)
    if family is None:
        return None
    family_cfg = _FAMILY_CONFIGS[family]
    return CalEventSpec(
        obsid=obsid_spec.obsid,
        cal_product=obsid_spec.cal_product,
        trimmed_product=obsid_spec.trimmed_product,
        family=family,
        companion_products=family_cfg.companion_products,
        time_variable=CAL_PRODUCT_TIME_VARIABLE,
    )


def get_cal_event_spec(obsid: int) -> CalEventSpec:
    """Return the rad cal-combine spec for a RAD ObsID.

    Parameters
    ----------
    obsid : int
        Radiometer calibration ObsID (``ICIE__SW_OBSID_RAD``).

    Returns
    -------
    CalEventSpec
        Event specification for cal-combine.

    Raises
    ------
    KeyError
        If the ObsID is unknown in ``libera_utils.obsids``.
    ValueError
        If the ObsID is known but not yet supported by rad cal-combine.
    """
    obsid_spec = get_obsid_spec(NomHkObsidSource.RAD, obsid)
    event = _cal_event_from_obsid_spec(obsid_spec)
    if event is None:
        raise ValueError(
            f"RAD ObsID {obsid} is not supported by libera_rad cal-combine (cal_product={obsid_spec.cal_product})"
        )
    return event


def _build_cal_event_by_obsid() -> dict[int, CalEventSpec]:
    """Derive supported rad cal events from the shared ObsID registry."""
    events: dict[int, CalEventSpec] = {}
    for obsid_spec in iter_trim_eligible(NomHkObsidSource.RAD):
        if obsid_spec.kind is not ObsIdKind.RAD_CAL:
            continue
        event = _cal_event_from_obsid_spec(obsid_spec)
        if event is not None:
            events[event.obsid] = event
    return events


#: Supported rad cal-combine ObsIDs, derived from ``libera_utils.obsids``.
CAL_EVENT_BY_OBSID: dict[int, CalEventSpec] = _build_cal_event_by_obsid()


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
