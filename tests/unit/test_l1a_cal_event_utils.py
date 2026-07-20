"""Unit tests for L1A calibration event utilities."""

import numpy as np
import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners import l1a_cal_event_utils as utils
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID


class TestSliceDatasetToTimeWindow:
    """Tests for slice_dataset_to_time_window."""

    @pytest.fixture
    def packet_dataset(self):
        times = np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00"],
            dtype="datetime64[s]",
        )
        return xr.Dataset(
            {
                "ICIE__SW_OBSID_RAD": ("PACKET", np.array([1, 2, 3], dtype=np.int32)),
            },
            coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        )

    def test_slices_packets_inclusive(self, packet_dataset):
        t0 = np.datetime64("2025-01-01T00:00:00", "s")
        t1 = np.datetime64("2025-01-01T00:01:00", "s")
        sliced = utils.slice_dataset_to_time_window(packet_dataset, t0, t1)
        assert sliced.sizes["PACKET"] == 2
        assert list(sliced["ICIE__SW_OBSID_RAD"].values) == [1, 2]

    def test_slices_secondary_time_dimension_independently(self):
        pkt_times = np.array(["2025-01-01T00:00:00", "2025-01-01T00:02:00"], dtype="datetime64[s]")
        fpe_times = np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:03:00"],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {"SIGNAL": (["PACKET", "RAD_SAMPLE_FPE_TIME"], np.ones((2, 3)))},
            coords={
                "PACKET_ICIE_TIME": ("PACKET", pkt_times),
                "RAD_SAMPLE_FPE_TIME": ("RAD_SAMPLE_FPE_TIME", fpe_times),
            },
        )
        sliced = utils.slice_dataset_to_time_window(
            ds,
            np.datetime64("2025-01-01T00:00:00", "s"),
            np.datetime64("2025-01-01T00:01:30", "s"),
            secondary_time_dim="RAD_SAMPLE_FPE_TIME",
        )
        assert sliced.sizes["PACKET"] == 1
        assert sliced.sizes["RAD_SAMPLE_FPE_TIME"] == 2


class TestSelectAndSliceEventInputs:
    """Tests for select_and_slice_event_inputs."""

    def test_selects_companions_and_slices_fpe_dims(self):
        event_spec = CAL_EVENT_BY_OBSID[512]
        times = np.array(
            [
                "2025-01-01T00:00:00",
                "2025-01-01T00:01:00",
                "2025-01-01T00:02:00",
                "2025-01-01T00:03:00",
            ],
            dtype="datetime64[ns]",
        )
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([512, 512], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times[1:3])},
        )
        rad_full = xr.Dataset(
            {"RAD_SIGNAL": (("PACKET", "RAD_FULL_FPE_TIME"), np.ones((4, 4)))},
            coords={
                "PACKET_ICIE_TIME": ("PACKET", times),
                "RAD_FULL_FPE_TIME": ("RAD_FULL_FPE_TIME", times),
            },
        )
        cal_full = xr.Dataset(
            {"CAL_SIGNAL": (("PACKET", "CAL_FULL_FPE_TIME"), np.ones((4, 4)))},
            coords={
                "PACKET_ICIE_TIME": ("PACKET", times),
                "CAL_FULL_FPE_TIME": ("CAL_FULL_FPE_TIME", times),
            },
        )
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-TRIMMED_V5-8-5_20250101T000100_20250101T000200_R26100000001.nc": nom_hk,
            "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": rad_full,
            "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": cal_full,
            "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": xr.Dataset(),
        }

        inputs = utils.select_and_slice_event_inputs(all_data, event_spec)
        assert len(inputs) == 3
        assert inputs[0] is nom_hk
        assert inputs[1].sizes["PACKET"] == 2
        assert inputs[1].sizes["RAD_FULL_FPE_TIME"] == 2
        assert inputs[2].sizes["PACKET"] == 2
        assert inputs[2].sizes["CAL_FULL_FPE_TIME"] == 2

    def test_raises_when_companion_missing(self):
        event_spec = CAL_EVENT_BY_OBSID[512]
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([512], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]"))},
        )
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-TRIMMED_V5-8-5_20250101T000000_20250101T000000_R26100000001.nc": nom_hk,
        }
        with pytest.raises(ValueError, match=DataProductIdentifier.l1a_icie_rad_full_decoded.value):
            utils.select_and_slice_event_inputs(all_data, event_spec)

    def test_uses_provided_nom_hk_without_reextract(self):
        event_spec = CAL_EVENT_BY_OBSID[384]
        times = np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00"],
            dtype="datetime64[ns]",
        )
        full_nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([2, 384, 384], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        )
        filtered = full_nom_hk.isel(PACKET=[1, 2])
        pev = xr.Dataset(
            {"X": ("PACKET", np.arange(3))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        )
        rad = xr.Dataset(
            {"Y": (("PACKET", "RAD_SAMPLE_FPE_TIME"), np.zeros((3, 3)))},
            coords={
                "PACKET_ICIE_TIME": ("PACKET", times),
                "RAD_SAMPLE_FPE_TIME": ("RAD_SAMPLE_FPE_TIME", times),
            },
        )
        all_data = {
            (
                "LIBERA_L1A_NOM-HK-SOLAR-SSW-PRI-TRIMMED_V5-8-5_20250101T000100_20250101T000200_R26100000001.nc"
            ): full_nom_hk,
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5_20250101T000000_20250101T000200_R26100000001.nc": pev,
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5_20250101T000000_20250101T000200_R26100000001.nc": rad,
        }
        inputs = utils.select_and_slice_event_inputs(all_data, event_spec, nom_hk=filtered)
        assert inputs[0] is filtered
        assert inputs[1].sizes["PACKET"] == 2
        assert inputs[1].sizes["RAD_SAMPLE_FPE_TIME"] == 2
        assert inputs[2].sizes["PACKET"] == 2
