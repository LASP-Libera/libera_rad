"""Unit tests for ObsID-dispatched cal_combine helpers."""

import numpy as np
import pytest
import xarray as xr

from libera_rad.calibration.combiners.cal_combine import resolve_cal_obsid_from_env
from libera_rad.calibration.combiners.l1a_cal_event_utils import confirm_obsid_matches_hk
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID, LIBERA_CAL_OBSID_ENV
from libera_rad.config import get_cal_product_definition


class TestResolveCalObsidFromEnv:
    """Tests for LIBERA_CAL_OBSID parsing."""

    def test_reads_valid_obsid(self, monkeypatch):
        monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, "512")
        assert resolve_cal_obsid_from_env() == 512

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv(LIBERA_CAL_OBSID_ENV, raising=False)
        with pytest.raises(ValueError, match=LIBERA_CAL_OBSID_ENV):
            resolve_cal_obsid_from_env()

    def test_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, "not-an-int")
        with pytest.raises(ValueError, match="must be an integer"):
            resolve_cal_obsid_from_env()

    def test_unknown_obsid_raises(self, monkeypatch):
        monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, "999")
        with pytest.raises(ValueError, match="Unknown calibration ObsID"):
            resolve_cal_obsid_from_env()


class TestConfirmObsidMatchesHk:
    """Tests for fail-closed HK ObsID confirmation."""

    def test_accepts_matching_obsid(self):
        nom_hk = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array([512, 512], dtype=np.int32))})
        confirm_obsid_matches_hk(nom_hk, 512)

    def test_accepts_filler_obsids_with_expected_cal(self):
        nom_hk = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array([2, 512, 2], dtype=np.int32))})
        confirm_obsid_matches_hk(nom_hk, 512)

    def test_rejects_missing_expected_obsid(self):
        nom_hk = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array([2, 3], dtype=np.int32))})
        with pytest.raises(ValueError, match="confirmation failed"):
            confirm_obsid_matches_hk(nom_hk, 512)

    def test_rejects_additional_cal_obsid(self):
        nom_hk = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array([320, 321], dtype=np.int32))})
        with pytest.raises(ValueError, match="additional calibration ObsIDs"):
            confirm_obsid_matches_hk(nom_hk, 320)


class TestGetCalProductDefinition:
    """Tests for family product-definition loading with ProductID override."""

    def test_product_id_matches_event_spec(self):
        for obsid, spec in (
            (512, CAL_EVENT_BY_OBSID[512]),
            (257, CAL_EVENT_BY_OBSID[257]),
            (320, CAL_EVENT_BY_OBSID[320]),
            (389, CAL_EVENT_BY_OBSID[389]),
        ):
            definition = get_cal_product_definition(spec)
            assert definition.attributes["ProductID"] == spec.cal_product.value
