"""Unit tests for L1A calibration event utilities."""

import numpy as np
import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners import l1a_cal_event_utils as utils
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID


def make_sample_companion(
    group: str,
    packet_times: np.ndarray,
    samples_per_packet: int = 2,
    sample_times: np.ndarray | None = None,
    with_packet_index: bool = True,
) -> xr.Dataset:
    """Build a decoded companion in real product shape: samples on their own axis.

    Real decoded L1A products put sample data on a 1-D ``<group>_FPE_TIME`` axis laid out
    packet-major, with a ``<group>_packet_index`` variable recording which packet each sample
    came from. They never carry 2-D (packet, sample) variables.

    Parameters
    ----------
    group : str
        Sample group prefix, e.g. ``"RAD_FULL"``.
    packet_times : np.ndarray
        Packet timestamps, one per packet.
    samples_per_packet : int
        Samples each packet expands to.
    sample_times : np.ndarray or None
        Sample timestamps. Defaults to tracking packet time so the two clocks agree.
    with_packet_index : bool
        Whether to include the ``<group>_packet_index`` variable, as real products do.

    Returns
    -------
    xr.Dataset
        Decoded-shaped companion Dataset.
    """
    sample_dim = f"{group}_FPE_TIME"
    n_packets = packet_times.size
    if sample_times is None:
        sample_times = np.repeat(packet_times, samples_per_packet)
    data_vars = {f"{group}_SIGNAL": (sample_dim, np.arange(n_packets * samples_per_packet, dtype=np.float32))}
    if with_packet_index:
        data_vars[f"{group}_packet_index"] = (
            sample_dim,
            np.repeat(np.arange(n_packets, dtype=np.int64), samples_per_packet),
        )
    return xr.Dataset(
        data_vars,
        coords={
            "PACKET_ICIE_TIME": ("PACKET", packet_times),
            sample_dim: (sample_dim, sample_times),
        },
    )


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
        rad_full = make_sample_companion("RAD_FULL", times, samples_per_packet=2)
        cal_full = make_sample_companion("CAL_FULL", times, samples_per_packet=2)
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-FAMILY-TRIMMED_V5-8-5_20250101T000100_20250101T000200_R26100000001.nc": nom_hk,
            "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": rad_full,
            "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": cal_full,
            "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-8-5_20250101T000000_20250101T000300_R26100000001.nc": xr.Dataset(),
        }

        inputs, input_file_names = utils.select_and_slice_event_inputs(all_data, event_spec)
        assert len(inputs) == 3
        assert inputs[0] is nom_hk
        # 2 packets inside the NOM-HK window, each contributing both of its samples.
        assert inputs[1].sizes["PACKET"] == 2
        assert inputs[1].sizes["RAD_FULL_FPE_TIME"] == 4
        assert inputs[2].sizes["PACKET"] == 2
        assert inputs[2].sizes["CAL_FULL_FPE_TIME"] == 4

        # Only the products actually used are reported, NOM-HK first; the unused PEC-SW-STAT
        # file on the manifest is not an input to a gain product.
        assert [name.split("_")[2] for name in input_file_names] == [
            "NOM-HK-GAIN-FAMILY-TRIMMED",
            "RAD-FULL-DECODED",
            "CAL-FULL-DECODED",
        ]

    def test_packet_index_is_renumbered_from_zero(self):
        """Trimmed companions must carry a 0-based index into their own packet axis."""
        event_spec = CAL_EVENT_BY_OBSID[512]
        times = np.arange(
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-01T00:00:10", "ns"),
            np.timedelta64(1, "s"),
        )
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.full(3, 512, dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times[4:7])},
        )
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-FAMILY-TRIMMED_V5-8-5_20250101T000004_20250101T000006_R26100000001.nc": nom_hk,
            "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5_20250101T000000_20250101T000009_R26100000001.nc": (
                make_sample_companion("RAD_FULL", times, samples_per_packet=3)
            ),
            "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5_20250101T000000_20250101T000009_R26100000001.nc": (
                make_sample_companion("CAL_FULL", times, samples_per_packet=3)
            ),
        }

        inputs, _ = utils.select_and_slice_event_inputs(all_data, event_spec)
        rad_full = inputs[1]
        assert rad_full.sizes["PACKET"] == 3
        # Source packets 4-6 become 0-2; the source values were 12-20
        np.testing.assert_array_equal(rad_full["RAD_FULL_packet_index"].values, np.repeat(np.arange(3), 3))
        np.testing.assert_array_equal(rad_full["RAD_FULL_SIGNAL"].values, np.arange(12, 21))

    def test_selection_follows_sample_time_not_packet_time(self):
        """Sample time drives which packets are kept, because samples are the science data."""
        event_spec = CAL_EVENT_BY_OBSID[512]
        times = np.arange(
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-01T00:00:06", "ns"),
            np.timedelta64(1, "s"),
        )
        # RAD-FULL's sample clock runs 2 s ahead of its packet clock
        skewed_samples = np.repeat(times + np.timedelta64(2, "s"), 2)
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.full(2, 512, dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times[4:6])},
        )
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-FAMILY-TRIMMED_V5-8-5_20250101T000004_20250101T000005_R26100000001.nc": nom_hk,
            "LIBERA_L1A_RAD-FULL-DECODED_V5-8-5_20250101T000000_20250101T000005_R26100000001.nc": (
                make_sample_companion("RAD_FULL", times, samples_per_packet=2, sample_times=skewed_samples)
            ),
            "LIBERA_L1A_CAL-FULL-DECODED_V5-8-5_20250101T000000_20250101T000005_R26100000001.nc": (
                make_sample_companion("CAL_FULL", times, samples_per_packet=2)
            ),
        }

        inputs, _ = utils.select_and_slice_event_inputs(all_data, event_spec)
        # Packet time would have kept source packets 4-5; their samples sit at 6-7 s, outside the
        # window. The packets whose samples land in [4 s, 5 s] are 2-3.
        np.testing.assert_array_equal(inputs[1]["RAD_FULL_SIGNAL"].values, np.arange(4, 8))
        # The unskewed companion is unaffected
        np.testing.assert_array_equal(inputs[2]["CAL_FULL_SIGNAL"].values, np.arange(8, 12))

    def test_raises_when_companion_missing(self):
        event_spec = CAL_EVENT_BY_OBSID[512]
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([512], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]"))},
        )
        all_data = {
            "LIBERA_L1A_NOM-HK-GAIN-FAMILY-TRIMMED_V5-8-5_20250101T000000_20250101T000000_R26100000001.nc": nom_hk,
        }
        with pytest.raises(ValueError, match=DataProductIdentifier.l1a_icie_rad_full_decoded.value):
            utils.select_and_slice_event_inputs(all_data, event_spec)

    def test_solar_event_window_comes_from_its_own_nom_hk_granule(self):
        event_spec = CAL_EVENT_BY_OBSID[384]
        times = np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00"],
            dtype="datetime64[ns]",
        )
        nom_hk = xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([384, 384], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times[1:])},
        )
        pev = xr.Dataset(
            {"X": ("PACKET", np.arange(3))},
            coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        )
        rad = make_sample_companion("RAD_SAMPLE", times, samples_per_packet=2)
        all_data = {
            ("LIBERA_L1A_NOM-HK-SOLAR-FAMILY-TRIMMED_V5-8-5_20250101T000100_20250101T000200_R26100000001.nc"): nom_hk,
            "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-8-5_20250101T000000_20250101T000200_R26100000001.nc": pev,
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5_20250101T000000_20250101T000200_R26100000001.nc": rad,
        }
        inputs, input_file_names = utils.select_and_slice_event_inputs(all_data, event_spec)
        assert inputs[0] is nom_hk
        assert inputs[1].sizes["PACKET"] == 2
        assert inputs[1].sizes["RAD_SAMPLE_FPE_TIME"] == 4
        assert inputs[2].sizes["PACKET"] == 2
        assert "NOM-HK-SOLAR-FAMILY-TRIMMED" in input_file_names[0]


class TestExtractNamedNomHkDataset:
    """One trimmed NOM-HK granule per event; anything else is a malformed manifest."""

    @staticmethod
    def _nom_hk(obsid: int) -> xr.Dataset:
        return xr.Dataset(
            {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([obsid], dtype=np.int32))},
            coords={"PACKET_ICIE_TIME": ("PACKET", np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]"))},
        )

    def test_returns_the_single_family_granule(self):
        event_spec = CAL_EVENT_BY_OBSID[385]
        name = "LIBERA_L1A_NOM-HK-SOLAR-FAMILY-TRIMMED_V5-8-5_20250101T000000_20250101T000100_R26100000001.nc"
        all_data = {name: self._nom_hk(385)}

        assert utils.extract_named_nom_hk_dataset(all_data, event_spec) == (name, all_data[name])

    def test_two_family_granules_raise_and_name_both(self):
        """A family ProductID covers several ObsIDs, so it cannot pick the event's granule."""
        event_spec = CAL_EVENT_BY_OBSID[385]
        first = "LIBERA_L1A_NOM-HK-SOLAR-FAMILY-TRIMMED_V5-8-5_20250101T000000_20250101T000100_R26100000001.nc"
        second = "LIBERA_L1A_NOM-HK-SOLAR-FAMILY-TRIMMED_V5-8-5_20250101T010000_20250101T010100_R26100000001.nc"
        all_data = {first: self._nom_hk(385), second: self._nom_hk(386)}

        with pytest.raises(ValueError, match="expects one NOM-HK granule per calibration event") as exc:
            utils.extract_named_nom_hk_dataset(all_data, event_spec)
        assert first in str(exc.value)
        assert second in str(exc.value)

    def test_full_day_decoded_nom_hk_is_not_a_substitute(self):
        """It spans many ObsIDs, so it would give a day-wide window instead of the event's."""
        event_spec = CAL_EVENT_BY_OBSID[385]
        name = "LIBERA_L1A_NOM-HK-DECODED_V5-8-5_20250101T000000_20250102T000000_R26100000001.nc"
        all_data = {name: self._nom_hk(385)}

        with pytest.raises(ValueError, match="No NOM-HK-SOLAR-FAMILY-TRIMMED granule"):
            utils.extract_named_nom_hk_dataset(all_data, event_spec)

    def test_no_nom_hk_at_all_raises(self):
        event_spec = CAL_EVENT_BY_OBSID[385]
        all_data = {
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5_20250101T000000_20250101T000100_R26100000001.nc": xr.Dataset(),
        }

        with pytest.raises(ValueError, match="No NOM-HK-SOLAR-FAMILY-TRIMMED granule"):
            utils.extract_named_nom_hk_dataset(all_data, event_spec)
