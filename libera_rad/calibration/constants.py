"""Constants for libera_rad calibration and L1B radiance processing.

ObsID → CAL/TRIMMED ProductIDs and calibration dependency families live in
``libera_utils.obsids``. A family is identified by its TRIMMED ProductID; this
module owns only the rad-side merge recipe for each one (which companion streams
are merged into the product), plus combiner CCSDS field lists and L1B science enums.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import NomHkObsidSource, ObsIdSpec, get_family_specs, get_obsid_spec

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
#: NOM-HK defines the calibration event — the window comes from its packet time and every
#: companion is trimmed to it — so it is the only time base shared by all families. Naming a
#: product from a companion stamps one stream's extent rather than the event's, which can be
#: several minutes short of the data in the file. One constant rather than a per-family
#: setting, so a new family cannot opt into that.
#:
#: The window is not a strict envelope of the merged product. Companions are trimmed a whole
#: packet at a time, so an edge packet extends past the window by up to its own sample span,
#: and an FPE-skewed packet can extend further (a RAD-SAMPLE packet stamped 16 s past the
#: window while its samples land inside it has been observed in ground-test data).
CAL_PRODUCT_TIME_VARIABLE = "NOM_HK_PACKET_ICIE_TIME"

#: Companion L1A products merged into each family's calibration product, keyed by the family
#: TRIMMED ProductID from ``libera_utils.obsids``.
#:
#: This is the rad merge recipe, not the family's deployed input set. ``get_family_inputs``
#: declares what libera_cdk stages on the manifest, which is a superset: SWC/LWC/SOLAR also
#: stage AXIS-SAMPLE, the encoder source the AZROT/ELSCAN motor CKs are built from. cal-combine
#: does not build them yet — it requires the finished CKs on the manifest and ignores a staged
#: AXIS-SAMPLE granule. Every entry here must appear in ``get_family_inputs`` for its family.
_FAMILY_COMPANIONS: dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]] = {
    DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed: (
        DataProductIdentifier.l1a_icie_rad_full_decoded,
        DataProductIdentifier.l1a_icie_cal_full_decoded,
    ),
    DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed: (
        DataProductIdentifier.l1a_icie_cal_sample_decoded,
        DataProductIdentifier.l1a_icie_rad_sample_decoded,
        DataProductIdentifier.l1a_pec_sw_stat_decoded,
        DataProductIdentifier.l1a_pev_sw_stat_decoded,
    ),
    DataProductIdentifier.l1a_icie_nom_hk_lwc_family_trimmed: (
        DataProductIdentifier.l1a_icie_rad_sample_decoded,
        DataProductIdentifier.l1a_pec_sw_stat_decoded,
        DataProductIdentifier.l1a_pev_sw_stat_decoded,
    ),
    DataProductIdentifier.l1a_icie_nom_hk_solar_family_trimmed: (
        DataProductIdentifier.l1a_icie_rad_sample_decoded,
        DataProductIdentifier.l1a_pev_sw_stat_decoded,
    ),
    # TODO [LIBSDC-811]: Add lunar cals
}

#: Calibration families rad cal-combine implements. A registry family without an entry here
#: (lunar) is unsupported and its ObsIDs stay out of CAL_EVENT_BY_OBSID.
SUPPORTED_CAL_FAMILIES: frozenset[DataProductIdentifier] = frozenset(_FAMILY_COMPANIONS)


@dataclass(frozen=True)
class CalEventSpec:
    """Specification for one ObsID-specific calibration combine event."""

    obsid: int
    cal_product: DataProductIdentifier
    #: Family TRIMMED ProductID; the family key everything downstream dispatches on.
    trimmed_product: DataProductIdentifier
    companion_products: tuple[DataProductIdentifier, ...]
    #: Time coordinate the product filename is stamped from; always
    #: :data:`CAL_PRODUCT_TIME_VARIABLE`.
    time_variable: str


def _cal_event_from_obsid_spec(obsid_spec: ObsIdSpec) -> CalEventSpec | None:
    """Build a ``CalEventSpec`` when the registry entry belongs to a supported family."""
    if obsid_spec.cal_product is None or obsid_spec.trimmed_product is None:
        return None
    companions = _FAMILY_COMPANIONS.get(obsid_spec.trimmed_product)
    if companions is None:
        return None
    return CalEventSpec(
        obsid=obsid_spec.obsid,
        cal_product=obsid_spec.cal_product,
        trimmed_product=obsid_spec.trimmed_product,
        companion_products=companions,
        time_variable=CAL_PRODUCT_TIME_VARIABLE,
    )


def get_cal_event_spec(obsid: int) -> CalEventSpec:
    """Return the rad cal-combine spec for a RAD ObsID.

    The single gate from an ObsID number to a dispatchable event: callers hand it whatever
    they were given and get either a spec or an explanation.

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
    ValueError
        If the ObsID is not a RAD ObsID in ``libera_utils.obsids``, or is one whose family
        rad cal-combine does not implement. The two cases carry different messages.
    """
    try:
        obsid_spec = get_obsid_spec(NomHkObsidSource.RAD, obsid)
    except KeyError:
        raise ValueError(f"Unknown RAD ObsID {obsid}. cal-combine ObsIDs: {sorted(CAL_EVENT_BY_OBSID)}") from None
    event = _cal_event_from_obsid_spec(obsid_spec)
    if event is None:
        raise ValueError(
            f"RAD ObsID {obsid} ({obsid_spec.description}) is known but not supported by "
            f"libera_rad cal-combine (family={obsid_spec.trimmed_product}, "
            f"cal_product={obsid_spec.cal_product}). cal-combine ObsIDs: {sorted(CAL_EVENT_BY_OBSID)}"
        )
    return event


def _build_cal_event_by_obsid() -> dict[int, CalEventSpec]:
    """Expand each supported family into its member ObsIDs via the shared registry.

    Keyed by bare ObsID, unlike ``libera_utils.obsids.OBSID_REGISTRY`` which keys on
    ``(source, obsid)`` because the two NOM-HK ObsID namespaces collide (256 is SWC-365NM on
    RAD and Darks-of-Darks on WFOV). Dropping the source is only safe while every supported
    family is a RAD family, so that is checked rather than assumed.
    """
    events: dict[int, CalEventSpec] = {}
    for family_product in _FAMILY_COMPANIONS:
        for obsid_spec in get_family_specs(family_product):
            if obsid_spec.source is not NomHkObsidSource.RAD:
                raise ValueError(
                    f"{family_product.name!r} is a {obsid_spec.source.name} family; cal-combine "
                    f"indexes by bare ObsID and can only carry RAD families"
                )
            event = _cal_event_from_obsid_spec(obsid_spec)
            if event is not None:
                events[event.obsid] = event
    return events


#: Supported rad cal-combine ObsIDs, expanded from the supported families in
#: ``libera_utils.obsids``. One entry per ObsID; ObsIDs in the same family share a
#: TRIMMED product and a merge recipe but keep their own CAL product.
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
