"""Unit tests for ObsID-dispatched cal_algorithm helpers."""

import numpy as np
import pytest
import xarray as xr

from libera_rad.calibration.cal_algorithm import resolve_cal_obsid_from_env
from libera_rad.calibration.combiners.l1a_cal_event_utils import confirm_obsid_matches_hk
from libera_rad.calibration.constants import LIBERA_CAL_OBSID_ENV


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

    def test_unknown_obsid_is_not_rejected_here(self, monkeypatch):
        """Dispatchability is ``get_cal_event_spec``'s call, not this function's."""
        monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, "999")
        assert resolve_cal_obsid_from_env() == 999


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
