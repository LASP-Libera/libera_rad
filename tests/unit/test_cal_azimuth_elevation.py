"""Unit tests for calibration Azimuth_Position / Elevation_Position helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from libera_rad.calibration.combiners import l1a_cal_event_utils as utils
from libera_rad.l1b import REQUIRED_SPICE_CAL_AZEL


def _event_with_fpe(n: int = 4) -> xr.Dataset:
    times = np.arange(n).astype("datetime64[ns]")
    return xr.Dataset(
        {"ICIE__RAD_SAMPLE_0": ("RAD_SAMPLE_FPE_TIME", np.zeros(n, dtype=np.float32))},
        coords={"RAD_SAMPLE_FPE_TIME": ("RAD_SAMPLE_FPE_TIME", times)},
    )


class TestFamilyNeedsAzimuthElevationPositions:
    """Tests for family_needs_azimuth_elevation_positions."""

    @pytest.mark.parametrize(("family", "expected"), [("swc", True), ("lwc", True), ("solar", True), ("gain", False)])
    def test_families(self, family: str, expected: bool):
        assert utils.family_needs_azimuth_elevation_positions(family) is expected


class TestAttachAzimuthElevationPositions:
    """Tests for attach_azimuth_elevation_positions."""

    def test_use_geo_false_writes_fill(self):
        event = _event_with_fpe(5)
        result = utils.attach_azimuth_elevation_positions(event, [], use_geo=False)
        assert np.all(result["Azimuth_Position"].values == np.float32(-999))
        assert np.all(result["Elevation_Position"].values == np.float32(-999))
        assert result["Azimuth_Position"].dims == ("RAD_SAMPLE_FPE_TIME",)

    def test_use_geo_true_requires_kernels(self):
        event = _event_with_fpe()
        with pytest.raises(ValueError, match="SPICE kernel sources are required"):
            utils.attach_azimuth_elevation_positions(event, [], use_geo=True)

    def test_use_geo_true_calls_geolocation(self):
        event = _event_with_fpe(3)
        az = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        el = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_km = MagicMock()
        mock_km.__enter__ = MagicMock(return_value=mock_km)
        mock_km.__exit__ = MagicMock(return_value=False)

        with (
            patch("libera_rad.calibration.combiners.l1a_cal_event_utils.KernelManager", return_value=mock_km),
            patch(
                "libera_rad.calibration.combiners.l1a_cal_event_utils.geolocation.calculate_azimuth_elevation_for_timestamps",
                return_value=(az, el),
            ) as mock_calc,
        ):
            result = utils.attach_azimuth_elevation_positions(
                event, ["az.bc", "el.bc"], use_geo=True
            )

        mock_km.load_libera_dynamic_kernels.assert_called_once()
        mock_calc.assert_called_once()
        np.testing.assert_array_equal(result["Azimuth_Position"].values, az)
        np.testing.assert_array_equal(result["Elevation_Position"].values, el)

    def test_missing_fpe_time_raises(self):
        event = xr.Dataset({"x": ("PACKET", [1])})
        with pytest.raises(ValueError, match="RAD_SAMPLE_FPE_TIME"):
            utils.attach_azimuth_elevation_positions(event, [], use_geo=False)


class TestReadCalibrationManifestData:
    """Tests for read_calibration_manifest_data kernel intake."""

    def test_use_geo_false_returns_empty_kernels(self):
        manifest = MagicMock()
        manifest.configuration = {"use_geo": False}
        with patch(
            "libera_rad.calibration.combiners.l1a_cal_event_utils.read_all_input_data",
            return_value=({}, []),
        ) as mock_read:
            data, kernels = utils.read_calibration_manifest_data(manifest, require_azel_kernels=True)
        assert data == {}
        assert kernels == []
        mock_read.assert_called_once_with(manifest, required_spice=REQUIRED_SPICE_CAL_AZEL)

    def test_use_geo_true_requests_azel_kernels(self):
        manifest = MagicMock()
        manifest.configuration = {}
        with patch(
            "libera_rad.calibration.combiners.l1a_cal_event_utils.read_all_input_data",
            return_value=({"a.nc": xr.Dataset()}, ["az.bc", "el.bc"]),
        ) as mock_read:
            data, kernels = utils.read_calibration_manifest_data(manifest, require_azel_kernels=True)
        assert "a.nc" in data
        assert kernels == ["az.bc", "el.bc"]
        mock_read.assert_called_once_with(manifest, required_spice=REQUIRED_SPICE_CAL_AZEL)

    def test_gain_path_forces_use_geo_false(self):
        manifest = MagicMock()
        manifest.configuration = {"use_geo": True}
        copied = MagicMock()
        manifest.model_copy.return_value = copied
        with patch(
            "libera_rad.calibration.combiners.l1a_cal_event_utils.read_all_input_data",
            return_value=({}, []),
        ) as mock_read:
            utils.read_calibration_manifest_data(manifest, require_azel_kernels=False)
        manifest.model_copy.assert_called_once()
        mock_read.assert_called_once_with(copied)
