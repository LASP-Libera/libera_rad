"""Unit tests for L1A calibration event utilities."""

import numpy as np
import pytest
import xarray as xr

from libera_rad.calibration.combiners import l1a_cal_event_utils as utils

_VALID_L1A_NC = "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"
_VALID_RAD_SAMPLE_NC = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"


class TestParseLiberaFilenameTimes:
    """Tests for parse_libera_filename_times."""

    def test_parses_valid_libera_l1a_filename(self, tmp_path):
        path = tmp_path / _VALID_L1A_NC
        path.touch()
        result = utils.parse_libera_filename_times(path)
        assert result is not None
        t_start, t_end = result
        assert t_start == np.datetime64("2025-11-20T17:59:50", "s")
        assert t_end == np.datetime64("2025-11-20T19:05:49", "s")

    def test_returns_none_for_non_libera_filename(self, tmp_path):
        path = tmp_path / "not_a_libera_name.nc"
        path.touch()
        assert utils.parse_libera_filename_times(path) is None


class TestFilterFilesByTimeWindow:
    """Tests for filter_files_by_time_window."""

    @pytest.fixture
    def libera_file(self, tmp_path):
        path = tmp_path / _VALID_L1A_NC
        path.touch()
        return path

    def test_includes_file_overlapping_window(self, libera_file):
        windows = [(np.datetime64("2025-11-20T18:00:00", "s"), np.datetime64("2025-11-20T18:30:00", "s"))]
        selected = utils.filter_files_by_time_window([libera_file], windows)
        assert selected == [libera_file]

    def test_excludes_file_outside_window(self, libera_file):
        windows = [(np.datetime64("2026-01-01T00:00:00", "s"), np.datetime64("2026-01-01T01:00:00", "s"))]
        selected = utils.filter_files_by_time_window([libera_file], windows)
        assert selected == []

    def test_includes_unparseable_filename_conservatively(self, tmp_path):
        bad = tmp_path / "short_nom_hk.nc"
        bad.touch()
        windows = [(np.datetime64("2026-01-01T00:00:00", "s"), np.datetime64("2026-01-01T01:00:00", "s"))]
        selected = utils.filter_files_by_time_window([bad], windows)
        assert selected == [bad]


class TestDetectValueRuns:
    """Tests for detect_value_runs."""

    def test_finds_single_qualifying_run(self):
        field = np.array([1, 42, 42, 42, 1], dtype=np.int32)
        pkt_times = np.array(
            [
                "2025-01-01T00:00:00",
                "2025-01-01T00:00:01",
                "2025-01-01T00:00:02",
                "2025-01-01T00:00:03",
                "2025-01-01T00:00:04",
            ],
            dtype="datetime64[s]",
        )
        windows = utils.detect_value_runs(field, pkt_times, target_value=42, min_count=3)
        assert len(windows) == 1
        assert windows[0] == (pkt_times[1], pkt_times[3])

    def test_excludes_run_shorter_than_min_count(self):
        field = np.array([42, 42], dtype=np.int32)
        times = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[s]")
        assert utils.detect_value_runs(field, times, target_value=42, min_count=3) == []

    def test_finds_two_separate_runs(self):
        field = np.array([42] * 3 + [1] + [42] * 3, dtype=np.int32)
        base = np.datetime64("2025-01-01T00:00:00", "s")
        times = base + np.arange(7).astype("timedelta64[s]")
        windows = utils.detect_value_runs(field, times, target_value=42, min_count=3)
        assert len(windows) == 2
        assert windows[0] == (times[0], times[2])
        assert windows[1] == (times[4], times[6])


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


class TestOpenAndSortL1aFiles:
    """Tests for open_and_sort_l1a_files."""

    def test_single_file_sorted_by_packet_icie_time(self, test_l1a_cal_data_path):
        path = test_l1a_cal_data_path / "short_nom_hk.nc"
        ds = utils.open_and_sort_l1a_files([path])
        times = ds["PACKET_ICIE_TIME"].values
        assert np.all(times[:-1] <= times[1:])

    def test_multi_file_concatenation_sorts_unsorted_inputs(self, test_l1a_cal_data_path, tmp_path):
        """Later files with out-of-order PACKET_ICIE_TIME must still yield sorted output."""
        src = test_l1a_cal_data_path / "short_nom_hk.nc"
        base = xr.open_dataset(src).load()
        times = base["PACKET_ICIE_TIME"].values
        half = len(times) // 2

        first_half = base.isel(PACKET=slice(0, half))
        second_half = base.isel(PACKET=slice(half, None))

        file_a = tmp_path / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T180000_R26016183821.nc"
        file_b = tmp_path / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T180000_20251120T190549_R26016183822.nc"
        first_half.to_netcdf(file_a)
        second_half.to_netcdf(file_b)

        # Pass the chronologically second segment first so open_and_sort must reorder.
        ds = utils.open_and_sort_l1a_files([file_b, file_a])
        out_times = ds["PACKET_ICIE_TIME"].values
        assert np.all(out_times[:-1] <= out_times[1:])
        assert len(out_times) == len(times)

    def test_raises_when_mixed_product_types(self, test_l1a_cal_data_path, tmp_path):
        nom_hk_src = test_l1a_cal_data_path / "short_nom_hk.nc"
        rad_src = test_l1a_cal_data_path / "short_rad_sample.nc"
        nom_hk = tmp_path / _VALID_L1A_NC
        rad_sample = tmp_path / _VALID_RAD_SAMPLE_NC
        nom_hk.symlink_to(nom_hk_src)
        rad_sample.symlink_to(rad_src)
        with pytest.raises(ValueError, match="same L1A product type"):
            utils.open_and_sort_l1a_files([nom_hk, rad_sample])


class TestLoadL1aProduct:
    """Tests for load_l1a_product."""

    def test_loads_matching_product_name(self, test_l1a_cal_data_path, tmp_path):
        src = test_l1a_cal_data_path / "short_nom_hk.nc"
        named = tmp_path / _VALID_L1A_NC
        named.symlink_to(src)
        ds = utils.load_l1a_product(tmp_path, "NOM-HK-DECODED")
        assert ds.sizes["PACKET"] == 10

    def test_raises_when_no_files_match_token(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="NOM-HK-DECODED"):
            utils.load_l1a_product(tmp_path, "NOM-HK-DECODED")


class TestScanFilesForFieldWindows:
    """Tests for scan_files_for_field_windows."""

    def test_detects_constant_obsid_run_in_nom_hk(self, test_l1a_cal_data_path, tmp_path):
        src = test_l1a_cal_data_path / "short_nom_hk.nc"
        named = tmp_path / _VALID_L1A_NC
        named.symlink_to(src)
        windows = utils.scan_files_for_field_windows(
            tmp_path,
            product_name="NOM-HK-DECODED",
            target_values=[2],
            min_count=5,
        )
        assert len(windows) == 1
        t0, t1 = windows[0]
        assert t0 <= t1
