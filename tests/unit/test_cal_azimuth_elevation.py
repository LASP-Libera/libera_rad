"""Unit tests for calibration Azimuth_Position / Elevation_Position helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners import l1a_cal_event_utils as utils


def _event_with_fpe(n: int = 4) -> xr.Dataset:
    times = np.arange(n).astype("datetime64[ns]")
    return xr.Dataset(
        {"ICIE__RAD_SAMPLE_0": ("RAD_SAMPLE_FPE_TIME", np.zeros(n, dtype=np.float32))},
        coords={"RAD_SAMPLE_FPE_TIME": ("RAD_SAMPLE_FPE_TIME", times)},
    )


class TestAttachAzimuthElevationPositions:
    """Tests for attach_azimuth_elevation_positions."""

    def test_requires_kernels(self):
        event = _event_with_fpe()
        with pytest.raises(ValueError, match="SPICE kernel sources are required"):
            utils.attach_azimuth_elevation_positions(event, [])

    def test_calls_geolocation(self):
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
            result = utils.attach_azimuth_elevation_positions(event, ["az.bc", "el.bc"])

        mock_km.load_libera_dynamic_kernels.assert_called_once()
        mock_calc.assert_called_once()
        np.testing.assert_array_equal(result["Azimuth_Position"].values, az)
        np.testing.assert_array_equal(result["Elevation_Position"].values, el)

    def test_missing_fpe_time_raises(self):
        event = xr.Dataset({"x": ("PACKET", [1])})
        with pytest.raises(ValueError, match="RAD_SAMPLE_FPE_TIME"):
            utils.attach_azimuth_elevation_positions(event, ["az.bc", "el.bc"])


def _read_manifest(manifest):
    """Run read_all_cal_input_data with NetCDF opening and filename parsing stubbed out."""
    mock_ds = xr.Dataset({"x": ("PACKET", [1])})
    mock_handle = MagicMock()
    mock_handle.__enter__ = MagicMock(return_value=mock_handle)
    mock_handle.__exit__ = MagicMock(return_value=False)

    with (
        patch("libera_rad.calibration.combiners.l1a_cal_event_utils.smart_open", return_value=mock_handle),
        patch("libera_rad.calibration.combiners.l1a_cal_event_utils.xr.open_dataset") as mock_open,
        patch(
            "libera_rad.calibration.combiners.l1a_cal_event_utils.LiberaDataProductFilename.from_file_path"
        ) as mock_from_path,
    ):
        mock_open.return_value.load.return_value = mock_ds
        mock_fn = MagicMock()
        mock_fn.data_product_id = DataProductIdentifier.l1a_icie_nom_hk_decoded
        mock_from_path.return_value = mock_fn
        return utils.read_all_cal_input_data(manifest)


class TestReadAllCalInputData:
    """Tests for read_all_cal_input_data.

    cal-combine has no SPICE inputs: it generates its own motor CKs from AXIS-SAMPLE, so the
    manifest carries L1A granules only.
    """

    def test_loads_l1a_datasets(self):
        nc_names = [
            "LIBERA_L1A_NOM-HK-DECODED_V5-8-5_20200101T000000_20200101T010000_R1.nc",
            "LIBERA_L1A_AXIS-SAMPLE-DECODED_V5-8-5_20200101T000000_20200101T010000_R1.nc",
        ]
        manifest = MagicMock()
        manifest.files = [MagicMock(filename=name) for name in nc_names]

        data = _read_manifest(manifest)

        assert list(data) == nc_names

    def test_no_kernels_required(self):
        """The old intake failed closed when AZROT/ELSCAN were absent; that requirement is gone."""
        nc_name = "LIBERA_L1A_NOM-HK-DECODED_V5-8-5_20200101T000000_20200101T010000_R1.nc"
        manifest = MagicMock()
        manifest.files = [MagicMock(filename=nc_name)]

        data = _read_manifest(manifest)

        assert nc_name in data
