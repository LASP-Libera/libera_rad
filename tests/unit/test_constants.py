import xarray as xr

from libera_rad.calibration.constants import ChannelName, find_channel_variable, get_channel_name_enum


class TestFindChannelVariable:
    """Tests for _find_channel_variable function."""

    def test_find_channel_variable_found(self):
        """Test finding a channel variable that exists."""
        rad_data = xr.Dataset({
            "RADIOMETER_CH_1": (["time"], [1, 2, 3]),
            "RADIOMETER_CH_2": (["time"], [4, 5, 6])
        })

        result = find_channel_variable(rad_data, "1")
        assert result == "RADIOMETER_CH_1"

    def test_find_channel_variable_not_found(self):
        """Test when channel variable is not found."""
        rad_data = xr.Dataset({
            "RADIOMETER_CH_1": (["time"], [1, 2, 3])
        })

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
