import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import (
    NomHkObsidSource,
    ObsIdKind,
    get_family_inputs,
    get_family_specs,
    get_obsid_spec,
    iter_trim_eligible,
)

from libera_rad.calibration.constants import (
    CAL_EVENT_BY_OBSID,
    SUPPORTED_CAL_FAMILIES,
    ChannelName,
    find_channel_variable,
    get_cal_event_spec,
    get_channel_name_enum,
)
from libera_rad.config import CAL_FAMILY_PRODUCT_DEFINITIONS


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
        supported: set[int] = set()
        unsupported: set[int] = set()
        for obsid_spec in iter_trim_eligible(NomHkObsidSource.RAD):
            if obsid_spec.kind is not ObsIdKind.RAD_CAL or obsid_spec.cal_product is None:
                continue
            if obsid_spec.trimmed_product in SUPPORTED_CAL_FAMILIES:
                supported.add(obsid_spec.obsid)
            else:
                unsupported.add(obsid_spec.obsid)
        assert unsupported  # lunar entries exist in utils
        assert unsupported.isdisjoint(CAL_EVENT_BY_OBSID)
        assert set(CAL_EVENT_BY_OBSID) == supported

    def test_supported_families_cover_their_whole_utils_family(self):
        """Every ObsID utils puts in a supported family is dispatchable, with its own CAL product."""
        for family in SUPPORTED_CAL_FAMILIES:
            members = get_family_specs(family)
            assert members
            cal_products = {member.cal_product for member in members}
            assert len(cal_products) == len(members)  # one CAL product per ObsID
            for member in members:
                event = CAL_EVENT_BY_OBSID[member.obsid]
                assert event.trimmed_product is family
                assert event.cal_product is member.cal_product

    def test_merge_recipe_is_a_subset_of_the_deployed_family_inputs(self):
        """Anything merged must also be staged on the manifest by the family's cdk node.

        ``get_family_inputs`` is the deployed input set and is a superset: AXIS-SAMPLE reaches
        cal-combine as AZROT/ELSCAN CK kernels rather than as a merged companion.
        """
        for event in CAL_EVENT_BY_OBSID.values():
            staged = set(get_family_inputs(event.trimmed_product))
            assert set(event.companion_products) <= staged, event.trimmed_product

    def test_every_supported_family_has_a_product_definition(self):
        assert set(CAL_FAMILY_PRODUCT_DEFINITIONS) == set(SUPPORTED_CAL_FAMILIES)

    def test_noise_is_distinct_event_on_the_gain_family(self):
        """Noise cal (ObsID 515) is its own event/product but combines on the gain family."""
        gain = get_cal_event_spec(512)
        noise = get_cal_event_spec(515)

        # Distinct calibration event with its own CAL product, sharing the family TRIMMED product.
        assert noise.obsid != gain.obsid
        assert noise.cal_product is DataProductIdentifier.cal_noise
        assert noise.trimmed_product is gain.trimmed_product
        assert noise.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed

        # ...combined with gain's full-rate merge recipe.
        assert noise.companion_products == gain.companion_products
        assert noise.time_variable == gain.time_variable

    def test_all_five_lwc_temperatures_supported(self):
        """LWC ObsIDs 320-324 all resolve to the lwc family."""
        lwc_obsids = {320: "LWC-310K", 321: "LWC-320K", 322: "LWC-335K", 323: "LWC-300K", 324: "LWC-305K"}
        for obsid, product_value in lwc_obsids.items():
            spec = get_cal_event_spec(obsid)
            assert spec.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lwc_family_trimmed
            assert spec.cal_product.value == product_value
