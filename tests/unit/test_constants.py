import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import NomHkObsidSource, ObsIdKind, get_obsid_spec, iter_trim_eligible

from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    ChannelName,
    family_from_cal_product,
    find_channel_variable,
    get_cal_event_spec,
    get_channel_name_enum,
)


class TestFindChannelVariable:
    """Tests for _find_channel_variable function."""

    def test_find_channel_variable_found(self):
        """Test finding a channel variable that exists."""
        rad_data = xr.Dataset({"RADIOMETER_CH_1": (["time"], [1, 2, 3]), "RADIOMETER_CH_2": (["time"], [4, 5, 6])})

        result = find_channel_variable(rad_data, "1")
        assert result == "RADIOMETER_CH_1"

    def test_find_channel_variable_not_found(self):
        """Test when channel variable is not found."""
        rad_data = xr.Dataset({"RADIOMETER_CH_1": (["time"], [1, 2, 3])})

        result = find_channel_variable(rad_data, "9")
        assert result is None


class TestGetChannelNameEnum:
    """Tests for get_channel_name_enum function."""

    def test_get_channel_name_enum_shortwave(self):
        """Test conversion for shortwave channel."""
        result = get_channel_name_enum("sw")
        assert result == ChannelName.SHORTWAVE

    def test_get_channel_name_enum_longwave(self):
        """Test conversion for longwave channel."""
        result = get_channel_name_enum("lw")
        assert result == ChannelName.LONGWAVE

    def test_get_channel_name_enum_total(self):
        """Test conversion for total channel."""
        result = get_channel_name_enum("total")
        assert result == ChannelName.TOTAL

    def test_get_channel_name_enum_split_shortwave(self):
        """Test conversion for split shortwave channel."""
        result = get_channel_name_enum("ssw")
        assert result == ChannelName.SPLIT_SHORTWAVE

    def test_get_channel_name_enum_invalid(self):
        """Test with invalid channel name."""
        result = get_channel_name_enum("invalid")
        assert result is None


class TestCalEventRegistry:
    """Guards for family derivation from libera_utils ObsIDs."""

    def test_supported_events_match_utils_products(self):
        for obsid, spec in CAL_EVENT_BY_OBSID.items():
            utils_spec = get_obsid_spec(NomHkObsidSource.RAD, obsid)
            assert spec.cal_product is utils_spec.cal_product
            assert spec.trimmed_product is utils_spec.trimmed_product
            assert get_cal_event_spec(obsid) == spec

    def test_registry_excludes_unsupported_rad_cal(self):
        """Lunar (and similar) RAD_CAL ObsIDs stay in utils but are not cal-combine yet."""
        unsupported = []
        for obsid_spec in iter_trim_eligible(NomHkObsidSource.RAD):
            if obsid_spec.kind is not ObsIdKind.RAD_CAL or obsid_spec.cal_product is None:
                continue
            if family_from_cal_product(obsid_spec.cal_product) is None:
                unsupported.append(obsid_spec.obsid)
        assert unsupported  # lunar entries exist in utils
        assert set(unsupported).isdisjoint(CAL_EVENT_BY_OBSID)
        assert len(CAL_EVENT_BY_OBSID) == 22

    def test_family_from_cal_product(self):
        assert family_from_cal_product(DataProductIdentifier.cal_gain) == "gain"
        assert family_from_cal_product(DataProductIdentifier.cal_swc_405nm) == "swc"
        assert family_from_cal_product(DataProductIdentifier.cal_lwc_temp1) == "lwc"
        assert family_from_cal_product(DataProductIdentifier.cal_solar_tot_pri) == "solar"
        assert family_from_cal_product(DataProductIdentifier.cal_lunar_cal1) is None
