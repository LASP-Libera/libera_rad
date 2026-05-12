"""Unit tests for lw_cal_combiner module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.lw_cal_combiner import (
    get_lw_event_type,
    get_product_definition_for_lw_cal_event,
)
from libera_rad.config import cal_lw_cal_product_definitions


def _make_all_data(obsids: list[int]) -> dict:
    """Return a minimal all_data dict whose NOM HK dataset contains the given OBSID values.

    extract_input_dataset is patched in each test, so the dict contents are not
    inspected by the production code — only the NOM HK dataset returned by the
    patch matters.  The obsids list is stored as the ICIE__SW_OBSID_RAD variable.
    """
    nom_hk_ds = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array(obsids, dtype=np.int32))})
    return {"nom_hk_placeholder": nom_hk_ds}


def _patch_extract(nom_hk_ds: xr.Dataset):
    """Return a context manager that patches extract_input_dataset to return nom_hk_ds."""
    return patch(
        "libera_rad.calibration.lw_cal_combiner.extract_input_dataset",
        return_value=nom_hk_ds,
    )


class TestGetLwEventType:
    """Tests for get_lw_event_type."""

    @pytest.mark.parametrize(
        ("obsid", "expected_identifier"),
        [
            (320, DataProductIdentifier.cal_lw_cal_temp1_combined),
            (321, DataProductIdentifier.cal_lw_cal_temp2_combined),
            (322, DataProductIdentifier.cal_lw_cal_temp3_combined),
        ],
    )
    def test_returns_correct_identifier_for_each_obsid(self, obsid, expected_identifier):
        """Returns the correct DataProductIdentifier for each recognised OBSID (320/321/322)."""
        all_data = _make_all_data([obsid])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            result = get_lw_event_type(all_data)

        assert result == expected_identifier

    def test_single_obsid_repeated_across_packets_returns_one_identifier(self):
        """A single OBSID value repeated across many packets is still treated as one unique match."""
        all_data = _make_all_data([320, 320, 320])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            result = get_lw_event_type(all_data)

        assert result == DataProductIdentifier.cal_lw_cal_temp1_combined

    def test_raises_when_no_obsid_matches(self):
        """Raises ValueError when none of the OBSIDs correspond to a known LW cal event."""
        all_data = _make_all_data([999])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            with pytest.raises(ValueError, match="No longwave calibration events detected in input data"):
                get_lw_event_type(all_data)

    def test_raises_when_multiple_lw_cal_obsids_present(self):
        """Raises ValueError when more than one distinct LW cal OBSID is found in the data."""
        all_data = _make_all_data([320, 321])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            with pytest.raises(ValueError, match="More than one longwave calibration event input data"):
                get_lw_event_type(all_data)

    def test_error_message_includes_obsids_when_no_match(self):
        """The no-match ValueError message includes the detected OBSID values."""
        all_data = _make_all_data([999])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            with pytest.raises(ValueError, match="999"):
                get_lw_event_type(all_data)

    def test_non_lw_cal_obsids_alongside_valid_one_are_ignored(self):
        """Unrecognised OBSIDs present in the same dataset do not affect the result."""
        # OBSID 999 is not in the mapping; only 322 should count.
        all_data = _make_all_data([999, 322])
        with _patch_extract(all_data["nom_hk_placeholder"]):
            result = get_lw_event_type(all_data)

        assert result == DataProductIdentifier.cal_lw_cal_temp3_combined


class TestGetProductDefinitionForLwCalEvent:
    """Tests for get_product_definition_for_lw_cal_event."""

    @pytest.mark.parametrize(
        "identifier",
        [
            DataProductIdentifier.cal_lw_cal_temp1_combined,
            DataProductIdentifier.cal_lw_cal_temp2_combined,
            DataProductIdentifier.cal_lw_cal_temp3_combined,
        ],
    )
    def test_returns_path_for_each_valid_identifier(self, identifier):
        """Returns a Path object for every valid LW cal DataProductIdentifier."""
        result = get_product_definition_for_lw_cal_event(identifier)

        assert isinstance(result, Path)

    @pytest.mark.parametrize(
        "identifier",
        [
            DataProductIdentifier.cal_lw_cal_temp1_combined,
            DataProductIdentifier.cal_lw_cal_temp2_combined,
            DataProductIdentifier.cal_lw_cal_temp3_combined,
        ],
    )
    def test_returned_path_points_to_existing_yaml(self, identifier):
        """The returned Path must point to an existing product definition YAML file."""
        result = get_product_definition_for_lw_cal_event(identifier)

        assert result.exists(), f"Product definition file not found: {result}"
        assert result.suffix == ".yml"

    @pytest.mark.parametrize(
        ("identifier", "expected_filename"),
        [
            (
                DataProductIdentifier.cal_lw_cal_temp1_combined,
                "CAL_LW_CAL_TEMP1_product_definition.yml",
            ),
            (
                DataProductIdentifier.cal_lw_cal_temp2_combined,
                "CAL_LW_CAL_TEMP2_product_definition.yml",
            ),
            (
                DataProductIdentifier.cal_lw_cal_temp3_combined,
                "CAL_LW_CAL_TEMP3_product_definition.yml",
            ),
        ],
    )
    def test_returns_correct_yaml_for_each_identifier(self, identifier, expected_filename):
        """Each DataProductIdentifier maps to its own correctly named product definition file."""
        result = get_product_definition_for_lw_cal_event(identifier)

        assert result.name == expected_filename

    def test_raises_value_error_for_unknown_identifier(self):
        """Raises ValueError when the identifier is not in cal_lw_cal_product_definitions."""
        with patch(
            "libera_rad.calibration.lw_cal_combiner.cal_lw_cal_product_definitions",
            {},
        ):
            with pytest.raises(ValueError, match="No longwave calibration event detected in input manifest"):
                get_product_definition_for_lw_cal_event(DataProductIdentifier.cal_lw_cal_temp1_combined)

    def test_error_message_includes_identifier_value(self):
        """The ValueError message includes the .value of the unrecognized identifier."""
        identifier = DataProductIdentifier.cal_lw_cal_temp1_combined
        with patch(
            "libera_rad.calibration.lw_cal_combiner.cal_lw_cal_product_definitions",
            {},
        ):
            with pytest.raises(ValueError, match=identifier.value):
                get_product_definition_for_lw_cal_event(identifier)

    def test_each_identifier_maps_to_distinct_path(self):
        """Every valid identifier must resolve to a different product definition file."""
        paths = [get_product_definition_for_lw_cal_event(identifier) for identifier in cal_lw_cal_product_definitions]
        assert len(paths) == len(set(paths)), "Two or more identifiers resolved to the same product definition path"
