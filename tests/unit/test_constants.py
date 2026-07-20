import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    SOLAR_FACE_BASE_OBSIDS,
    ChannelName,
    find_channel_variable,
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


class TestCalEventByObsid:
    """Tests for the ObsID → CalEventSpec registry."""

    def test_gain_obsid(self):
        spec = CAL_EVENT_BY_OBSID[512]
        assert spec.cal_product == DataProductIdentifier.cal_gain
        assert spec.family == "gain"
        assert spec.trimmed_product == DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed

    def test_products_match_libera_utils_registry(self):
        """CAL/TRIMMED ProductIDs are owned by libera_utils.obsids."""
        from libera_utils.obsids import NomHkObsidSource, get_obsid_spec

        for obsid, spec in CAL_EVENT_BY_OBSID.items():
            utils_spec = get_obsid_spec(NomHkObsidSource.RAD, obsid)
            assert spec.cal_product is utils_spec.cal_product
            assert spec.trimmed_product is utils_spec.trimmed_product

    def test_swc_obsid(self):
        spec = CAL_EVENT_BY_OBSID[257]
        assert spec.cal_product == DataProductIdentifier.cal_swc_405nm
        assert spec.family == "swc"

    def test_lwc_obsids(self):
        assert CAL_EVENT_BY_OBSID[320].cal_product == DataProductIdentifier.cal_lwc_temp1
        assert CAL_EVENT_BY_OBSID[321].cal_product == DataProductIdentifier.cal_lwc_temp2
        assert CAL_EVENT_BY_OBSID[322].cal_product == DataProductIdentifier.cal_lwc_temp3

    def test_solar_obsids(self):
        assert CAL_EVENT_BY_OBSID[384].cal_product == DataProductIdentifier.cal_solar_ssw_pri
        assert CAL_EVENT_BY_OBSID[389].cal_product == DataProductIdentifier.cal_solar_tot_sec
        assert CAL_EVENT_BY_OBSID[395].cal_product == DataProductIdentifier.cal_solar_sw_ter

    def test_solar_face_base_obsids(self):
        assert SOLAR_FACE_BASE_OBSIDS == {1: 384, 2: 388, 3: 392}

    def test_registry_has_22_events(self):
        assert len(CAL_EVENT_BY_OBSID) == 22
