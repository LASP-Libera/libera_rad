import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.constants import (
    COMBINER_GAIN_OBSID_TO_PRODUCT_IDENTIFIER,
    COMBINER_LW_OBSID_TO_PRODUCT_IDENTIFIER,
    COMBINER_SOLAR_FACE_BASE_OBSIDS,
    COMBINER_SOLAR_OBSID_TO_PRODUCT_IDENTIFIER,
    COMBINER_SW_OBSID_TO_PRODUCT_IDENTIFIER,
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


class TestCombinerObsidMappings:
    """Tests for calibration combiner OBSID → product identifier maps."""

    def test_gain_obsid_maps_to_gain_combined(self):
        assert COMBINER_GAIN_OBSID_TO_PRODUCT_IDENTIFIER[512] == DataProductIdentifier.cal_gain_combined

    def test_sw_obsid_maps_to_sw_combined(self):
        assert COMBINER_SW_OBSID_TO_PRODUCT_IDENTIFIER[257] == DataProductIdentifier.cal_sw_combined

    def test_lw_obsids_map_to_temp_combined_products(self):
        assert COMBINER_LW_OBSID_TO_PRODUCT_IDENTIFIER[320] == DataProductIdentifier.cal_lw_temp1_combined
        assert COMBINER_LW_OBSID_TO_PRODUCT_IDENTIFIER[321] == DataProductIdentifier.cal_lw_temp2_combined
        assert COMBINER_LW_OBSID_TO_PRODUCT_IDENTIFIER[322] == DataProductIdentifier.cal_lw_temp3_combined

    def test_solar_face_obsids_map_to_expected_products(self):
        assert COMBINER_SOLAR_OBSID_TO_PRODUCT_IDENTIFIER[384] == DataProductIdentifier.cal_solar_face1_combined
        assert COMBINER_SOLAR_OBSID_TO_PRODUCT_IDENTIFIER[388] == DataProductIdentifier.cal_solar_face2_combined
        assert COMBINER_SOLAR_OBSID_TO_PRODUCT_IDENTIFIER[392] == DataProductIdentifier.cal_solar_face3_combined

    def test_solar_face_base_obsids(self):
        assert COMBINER_SOLAR_FACE_BASE_OBSIDS == {1: 384, 2: 388, 3: 392}
