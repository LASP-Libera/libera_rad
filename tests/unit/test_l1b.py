"""Tests for the l1b algorithm"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import spiceypy as sp
import xarray as xr
from libera_utils.io.manifest import Manifest

from libera_rad import l1b


class TestReadAllInputData:
    """Tests for read_all_input_data function."""

    @pytest.fixture
    def mock_manifest(self):
        """Create a mock manifest with test files."""
        manifest = Mock()
        file_info_1 = Mock()
        file_info_1.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc"
        file_info_2 = Mock()
        file_info_2.filename = "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc"
        file_info_3 = Mock()
        file_info_3.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R00000000000.bc"
        manifest.files = [file_info_1, file_info_2, file_info_3]
        return manifest

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock xarray dataset."""
        dataset = xr.Dataset(
            {
                "variable1": (["time"], np.array([1, 2, 3])),
                "variable2": (["time"], np.array([4, 5, 6])),
            }
        )
        return dataset

    def test_read_all_input_data_collects_dynamic_kernel_paths(self, mock_manifest, mock_dataset):
        """Manifest kernel paths are returned for KernelManager (no package-local spice directory)."""
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with (
            patch("libera_rad.l1b.smart_open", return_value=mock_file),
            patch("xarray.open_dataset") as mock_open_dataset,
        ):
            mock_open_dataset.return_value.load.return_value = mock_dataset

            all_data, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

            assert dynamic_kernel_sources == [
                "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R00000000000.bc"
            ]
            assert len(all_data) == 2

    def test_read_all_input_data_loads_netcdf_files(self, mock_manifest, mock_dataset):
        """Test that NetCDF files are loaded correctly."""
        # Create a mock file handle that xr.open_dataset can use
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with (
            patch("libera_rad.l1b.smart_open", return_value=mock_file),
            patch("xarray.open_dataset") as mock_open_dataset,
        ):
            mock_open_dataset.return_value.load.return_value = mock_dataset

            all_data, _ = l1b.read_all_input_data(mock_manifest)

            assert len(all_data) == 2  # Two NetCDF files
            assert "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc" in all_data
            assert "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc" in all_data

    def test_read_all_input_data_handles_file_not_found(self, mock_manifest):
        """Test error handling when file is not found."""
        # Create a mock file handle that raises FileNotFoundError when entered
        mock_file = Mock()
        mock_file.__enter__ = Mock(side_effect=FileNotFoundError("File not found"))
        mock_file.__exit__ = Mock(return_value=False)

        with patch("libera_rad.l1b.smart_open", return_value=mock_file):
            with pytest.raises(FileNotFoundError):
                l1b.read_all_input_data(mock_manifest)

    def test_read_all_input_data_warns_on_empty_data(self, caplog):
        """Test warning is logged when no data files are loaded."""
        manifest = Mock()
        file_info = Mock()
        file_info.filename = "test_kernel.bc"
        manifest.files = [file_info]

        all_data, dynamic_kernel_sources = l1b.read_all_input_data(manifest)

        assert len(all_data) == 0
        assert dynamic_kernel_sources == ["test_kernel.bc"]
        assert "No data files were loaded" in caplog.text

    def test_read_all_input_data_handles_exception_in_processing(self, mock_manifest):
        """Test error handling when processing fails."""
        # Create a mock file handle that works for context manager
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with (
            patch("libera_rad.l1b.smart_open", return_value=mock_file),
            patch("xarray.open_dataset", side_effect=Exception("Processing error")),
        ):
            with pytest.raises(Exception, match="Processing error"):
                l1b.read_all_input_data(mock_manifest)

    def test_read_all_input_data_filters_files_correctly(self, mock_dataset):
        """Test that different file types are handled correctly."""
        manifest = Mock()

        # Create a mix of file types
        nc_file = Mock()
        nc_file.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc"
        bc_file = Mock()
        bc_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R00000000000.bc"
        bsp_file = Mock()
        bsp_file.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R00000000000.bsp"

        manifest.files = [nc_file, bc_file, bsp_file]

        # Create a mock file handle
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with (
            patch("libera_rad.l1b.smart_open", return_value=mock_file),
            patch("xarray.open_dataset") as mock_open_dataset,
        ):
            mock_open_dataset.return_value.load.return_value = mock_dataset

            all_data, dynamic_kernel_sources = l1b.read_all_input_data(manifest)

            assert len(all_data) == 1
            assert nc_file.filename in all_data
            assert dynamic_kernel_sources == [bc_file.filename, bsp_file.filename]

    def test_read_all_input_data_use_geo_false_skips_spice_even_with_spice_files(self, mock_dataset):
        """use_geo=false in manifest configuration should skip SPICE files and return empty kernel list."""
        manifest = Mock()
        nc_file = Mock()
        nc_file.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R00000000000.nc"
        bc_file = Mock()
        bc_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R00000000000.bc"
        manifest.files = [nc_file, bc_file]
        manifest.configuration = {"use_geo": False}

        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with (
            patch("libera_rad.l1b.smart_open", return_value=mock_file),
            patch("xarray.open_dataset") as mock_open_dataset,
        ):
            mock_open_dataset.return_value.load.return_value = mock_dataset
            all_data, dynamic_kernel_sources = l1b.read_all_input_data(manifest)

        assert len(all_data) == 1
        assert nc_file.filename in all_data
        assert dynamic_kernel_sources == []


class TestExtractRadiometerDatasets:
    """Tests for _extract_radiometer_datasets function."""

    def test_extract_radiometer_datasets_success(self):
        """Test successful extraction of radiometer and housekeeping datasets."""
        rad_dataset = xr.Dataset({"rad_var": (["time"], np.array([1, 2, 3]))})
        hk_dataset = xr.Dataset({"hk_var": (["time"], np.array([4, 5, 6]))})

        all_input_data = {
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183821.nc": rad_dataset,
            "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183621.nc": hk_dataset,
        }

        rad_data, nom_hk_data = l1b._extract_radiometer_datasets(all_input_data)

        assert "rad_var" in rad_data
        assert "hk_var" in nom_hk_data

    def test_extract_radiometer_datasets_missing_rad_data(self):
        """Test error when radiometer data is missing."""
        all_input_data = {
            "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183621.nc": xr.Dataset()
        }

        with pytest.raises(ValueError, match="No radiometer sample data found"):
            l1b._extract_radiometer_datasets(all_input_data)

    def test_extract_radiometer_datasets_missing_hk_data(self):
        """Test error when housekeeping data is missing."""
        all_input_data = {
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183821.nc": xr.Dataset(
                {"var": (["time"], [1, 2, 3])}
            )
        }

        with pytest.raises(ValueError, match="No nominal housekeeping data found"):
            l1b._extract_radiometer_datasets(all_input_data)


class TestCalculateDataQualityFlags:
    """Tests for calculate_data_quality_flags function."""

    def test_calculate_data_quality_flags_zeros(self):
        """Test that quality flags are initialized to zeros."""
        result = l1b.calculate_data_quality_flags(100)

        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert np.all(result == 0)
        assert result.dtype == np.uint32

    def test_calculate_data_quality_flags_empty(self):
        """Test with zero length."""
        result = l1b.calculate_data_quality_flags(0)

        assert len(result) == 0


class TestProcessL1aToL1b:
    """Integration tests for process_l1a_to_l1b function."""

    @pytest.fixture
    def mock_input_data(self):
        """Create mock input data."""
        rad_data = xr.Dataset(
            {
                "RAD_SAMPLE_FPE_TIME": (["time"], np.arange(1000, dtype=np.float64)),
                "ICIE__RAD_SAMPLE_1": (["time"], np.random.rand(1000)),
            }
        )

        nom_hk_data = xr.Dataset(
            {
                "PACKET_ICIE_TIME": (["time"], np.arange(0, 1000, 10, dtype=np.float64)),
                "ICIE__FPE_TSCOPE_TEMP": (["time"], np.full(100, 25.0)),
            }
        )

        return {
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183821.nc": rad_data,
            "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183821.nc": nom_hk_data,
        }

    def test_process_l1a_to_l1b(self, mock_input_data):
        """Test full L1A to L1B processing pipeline."""
        dynamic_kernel_sources = ["/tmp/dummy.bc"]

        with (
            patch("libera_rad.radiometer.radiance._load_calibration_data") as mock_load_cal,
            patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels"),
            patch("libera_rad.radiometer.radiance.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
            patch("libera_rad.geolocation.calculate_lat_lon_altitude") as mock_geoloc,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_radiance,
        ):
            # Setup mocks
            mock_cal = Mock()
            channel_prop = Mock()
            channel_prop.channel_enum = "1"
            mock_cal.channels = {"sw": channel_prop}
            mock_load_cal.return_value = mock_cal

            mock_downsample.side_effect = lambda x: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)
            mock_geoloc.return_value = pd.DataFrame(
                {"lat": np.random.rand(100), "lon": np.random.rand(100), "alt": np.random.rand(100)}
            )
            mock_radiance.return_value = pd.Series(np.random.rand(100))

            result, dynamic_attributes = l1b.process_l1a_to_l1b(mock_input_data, dynamic_kernel_sources, use_geo=True)

            # Check result structure
            assert isinstance(result, dict)
            assert "radiometer_time" in result
            assert "Latitude" in result
            assert "Filtered_Radiance_SW" in result
            assert isinstance(dynamic_attributes, dict)
            assert "Earth_Sun_Distance_AU" in dynamic_attributes

    def test_process_l1a_to_l1b_use_geo_false(self, mock_input_data):
        """use_geo=False should bypass KernelManager and SPICE geolocation."""
        with (
            patch("libera_rad.radiometer.radiance._load_calibration_data") as mock_load_cal,
            patch("libera_rad.l1b.KernelManager") as mock_kernel_manager_cls,
            patch("libera_rad.radiometer.radiance.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
            patch("libera_rad.geolocation.create_placeholder_geolocation_dataframe") as mock_placeholder_geo,
            patch("libera_rad.geolocation.calculate_geolocation_for_timestamps") as mock_calculate_geo,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_radiance,
        ):
            mock_cal = Mock()
            channel_prop = Mock()
            channel_prop.channel_enum = "1"
            mock_cal.channels = {"sw": channel_prop}
            mock_load_cal.return_value = mock_cal

            mock_downsample.side_effect = lambda x: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)
            mock_placeholder_geo.return_value = pd.DataFrame(
                {
                    "lat": np.full(100, -999, dtype=np.float32),
                    "lon": np.full(100, -999, dtype=np.float32),
                    "alt": np.full(100, -9999, dtype=np.float32),
                }
            )
            mock_radiance.return_value = pd.Series(np.random.rand(100))

            result, dynamic_attributes = l1b.process_l1a_to_l1b(mock_input_data, [], use_geo=False)

        mock_kernel_manager_cls.assert_not_called()
        mock_calculate_geo.assert_not_called()
        mock_placeholder_geo.assert_called_once_with(100)
        assert isinstance(result, dict)
        assert "Latitude" in result
        assert np.all(result["Latitude"] == np.float32(-999))
        assert isinstance(dynamic_attributes, dict)

    def test_process_l1a_to_l1b_requires_kernel_sources_when_use_geo_true(self, mock_input_data):
        """use_geo=True requires SPICE kernel paths from the manifest."""
        timestamps = np.arange(10, dtype=np.float64)
        calibrated_data = {"sw": np.zeros(10)}

        with patch(
            "libera_rad.radiometer.radiance.calibrate_and_downsample_radiometer_data",
            return_value=(timestamps, calibrated_data),
        ):
            with pytest.raises(
                ValueError,
                match="SPICE kernel sources are required when use_geo is True",
            ):
                l1b.process_l1a_to_l1b(mock_input_data, [], use_geo=True)


class TestAlgorithm:
    """Tests for algorithm function."""

    @pytest.fixture(autouse=True)
    def clear_spice_state(self):
        """Clear SPICE kernel state between tests."""
        try:
            sp.kclear()
        except Exception("no spice kernels exist"):
            pass

        yield

        try:
            sp.kclear()
        except Exception("no spice kernels exist"):
            pass

    @pytest.fixture
    def mock_manifest_obj(self):
        """Create mock manifest object."""
        manifest = Mock()
        manifest.files = []
        return manifest

    def test_algorithm_missing_processing_path(self, tmp_path, monkeypatch):
        """Test error when PROCESSING_PATH is not set."""
        manifest_path = tmp_path / "manifest.json"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("libera_utils.Manifest.from_file") as mock_from_file,
            patch("libera_rad.l1b.read_all_input_data") as mock_read_all,
            patch("libera_rad.l1b.process_l1a_to_l1b"),
        ):
            mock_from_file.return_value = Mock(files=[manifest_path], configuration={})
            mock_read_all.return_value = ({"foo": xr.Dataset()}, tmp_path)

            with pytest.raises(ValueError, match="PROCESSING_PATH environment variable is not set"):
                l1b.algorithm(manifest_path)

    def test_algorithm(self, generate_input_manifest, monkeypatch, tmp_path):
        """Testing the algorithm to generate output manifests"""

        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = generate_input_manifest()

        # Run the algorithm
        output_manifest_path = l1b.algorithm(manifest_path)

        output_manifest_obj = Manifest.from_file(output_manifest_path)

        for file in output_manifest_obj.files:
            data_product = xr.open_dataset(file.filename)
            print(data_product)
