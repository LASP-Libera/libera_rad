"""Tests for the l1b algorithm"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import spiceypy as sp
import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.manifest import Manifest

from libera_rad import l1b
from libera_rad.constants import CLOCK_RATE_MIN_CONE_ANGLE_DEG, MOON_IN_FOV_FILL_VALUE


class TestRequireSpiceInputs:
    def test_require_spice_inputs_ok(self):
        spice_files = {product_id: f"/tmp/{product_id.name}.bc" for product_id in l1b._REQUIRED_SPICE_PRODUCTION}
        l1b._require_spice_inputs(spice_files, l1b._REQUIRED_SPICE_PRODUCTION)

    def test_require_spice_inputs_missing(self):
        with pytest.raises(ValueError, match="Input manifest missing required SPICE data products: JPSS-SPK"):
            l1b._require_spice_inputs({}, l1b._REQUIRED_SPICE_JPSS_ONLY)


class TestReadAllInputData:
    """Tests for read_all_input_data function."""

    @pytest.fixture
    def mock_manifest(self):
        """Create a mock manifest with test files."""
        manifest = Mock()
        file_info_1 = Mock()
        file_info_1.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc"
        file_info_2 = Mock()
        file_info_2.filename = "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc"
        file_info_3 = Mock()
        file_info_3.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        file_info_4 = Mock()
        file_info_4.filename = "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        file_info_5 = Mock()
        file_info_5.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bsp"
        file_info_6 = Mock()
        file_info_6.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bc"
        manifest.files = [file_info_1, file_info_2, file_info_3, file_info_4, file_info_5, file_info_6]
        manifest.configuration = {}
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
                "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc",
                "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc",
                "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bsp",
                "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bc",
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
            assert "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc" in all_data
            assert "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc" in all_data

    def test_read_all_input_data_handles_file_not_found(self, mock_manifest):
        """Test error handling when file is not found."""
        # Create a mock file handle that raises FileNotFoundError when entered
        mock_file = Mock()
        mock_file.__enter__ = Mock(side_effect=FileNotFoundError("File not found"))
        mock_file.__exit__ = Mock(return_value=False)

        with patch("libera_rad.l1b.smart_open", return_value=mock_file):
            with pytest.raises(FileNotFoundError):
                l1b.read_all_input_data(mock_manifest)

    def test_read_all_input_data_rejects_incomplete_spice_manifest(self):
        """Production mode requires four dynamic SPICE kernels before processing."""
        manifest = Mock()
        az_file = Mock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        el_file = Mock()
        el_file.filename = "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        manifest.files = [az_file, el_file]
        manifest.configuration = {}

        with pytest.raises(ValueError, match="Input manifest missing required SPICE"):
            l1b.read_all_input_data(manifest)

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
        nc_file.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc"
        az_file = Mock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        el_file = Mock()
        el_file.filename = "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        bsp_file = Mock()
        bsp_file.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bsp"
        jpss_ck = Mock()
        jpss_ck.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bc"

        manifest.files = [nc_file, az_file, el_file, bsp_file, jpss_ck]
        manifest.configuration = {}

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
            assert dynamic_kernel_sources == [
                az_file.filename,
                el_file.filename,
                bsp_file.filename,
                jpss_ck.filename,
            ]

    def test_read_all_input_data_use_geo_false_skips_spice(self, mock_dataset, caplog):
        """use_geo false should skip SPICE files and return an empty kernel list."""
        import logging

        manifest = Mock()
        nc_file = Mock()
        nc_file.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc"
        bc_file = Mock()
        bc_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        manifest.files = [nc_file, bc_file]
        manifest.configuration = {"use_geo": False}

        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with caplog.at_level(logging.WARNING):
            with (
                patch("libera_rad.l1b.smart_open", return_value=mock_file),
                patch("xarray.open_dataset") as mock_open_dataset,
            ):
                mock_open_dataset.return_value.load.return_value = mock_dataset
                all_data, dynamic_kernel_sources = l1b.read_all_input_data(manifest)

        assert len(all_data) == 1
        assert nc_file.filename in all_data
        assert dynamic_kernel_sources == []
        assert "use_geo is false: skipping SPICE kernel" in caplog.text

    def test_read_all_input_data_jpss_only_mode_filters_kernels(self, mock_dataset, caplog):
        """jpss_only collects only JPSS kernels and warns on motor kernels."""
        import logging

        manifest = Mock()
        nc_file = Mock()
        nc_file.filename = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26092192956.nc"
        az_file = Mock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26092192956.bc"
        jpss_spk = Mock()
        jpss_spk.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bsp"
        jpss_ck = Mock()
        jpss_ck.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bc"
        manifest.files = [nc_file, az_file, jpss_spk, jpss_ck]
        manifest.configuration = {"jpss_only": True}

        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        with caplog.at_level(logging.WARNING):
            with (
                patch("libera_rad.l1b.smart_open", return_value=mock_file),
                patch("xarray.open_dataset") as mock_open_dataset,
            ):
                mock_open_dataset.return_value.load.return_value = mock_dataset
                _, dynamic_kernel_sources = l1b.read_all_input_data(manifest)

        assert dynamic_kernel_sources == [jpss_spk.filename, jpss_ck.filename]
        assert "jpss_only mode: skipping SPICE file" in caplog.text


class TestExtractInputDataset:
    """Tests for extract_input_dataset function."""

    def test_extract_input_dataset_success(self):
        """Test successful extraction of radiometer and housekeeping datasets."""
        rad_dataset = xr.Dataset({"rad_var": (["time"], np.array([1, 2, 3]))})
        hk_dataset = xr.Dataset({"hk_var": (["time"], np.array([4, 5, 6]))})

        all_input_data = {
            "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183821.nc": rad_dataset,
            "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183621.nc": hk_dataset,
        }

        rad_data = l1b.extract_input_dataset(all_input_data, DataProductIdentifier.l1a_icie_rad_sample_decoded)
        nom_hk_data = l1b.extract_input_dataset(all_input_data, DataProductIdentifier.l1a_icie_nom_hk_decoded)

        assert "rad_var" in rad_data
        assert "hk_var" in nom_hk_data

    def test_extract_input_dataset_missing_data(self):
        """Test error raised when the requested dataset is missing."""
        all_input_data = {
            "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20201120T175950_20201120T190549_R26016183621.nc": xr.Dataset()
        }

        with pytest.raises(ValueError, match="No dataset found in input files: RAD-SAMPLE-DECODED"):
            l1b.extract_input_dataset(all_input_data, DataProductIdentifier.l1a_icie_rad_sample_decoded)


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


def test_gate_clock_rate_near_nadir():
    # The clock angle is an azimuth about nadir, so its rate is undefined there. Samples
    # inside the cone gate are filled; samples at or beyond it are kept; and a coverage gap
    # (NaN cone) stays NaN rather than being turned into a fill value. Cone angles are taken
    # from the gate constant so the boundary case stays on the boundary if the gate moves.
    gate = float(CLOCK_RATE_MIN_CONE_ANGLE_DEG)
    rate = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)
    cone = np.array([gate / 2, gate, gate + 30.0, np.nan], dtype=np.float32)

    gated = l1b._gate_clock_rate_near_nadir(rate, cone)

    assert gated[0] == np.float32(-999.0)  # inside the nadir cone -> filled
    assert gated[1] == np.float32(2.0)  # exactly at the gate -> kept
    assert gated[2] == np.float32(3.0)
    assert np.isnan(gated[3])  # coverage gap stays NaN
    assert gated.dtype == np.float32


class TestProcessL1aToL1b:
    """Integration tests for process_l1a_to_l1b function."""

    @pytest.fixture
    def mock_input_data(self):
        """Create mock input data."""
        rad_times = pd.date_range("2025-01-01", periods=1000, freq="ms").values
        rad_data = xr.Dataset(
            {
                "RAD_SAMPLE_FPE_TIME": (["time"], rad_times),
                "PACKET_ICIE_TIME": (["time"], rad_times),
                "ICIE__RAD_OBSID_RAD": (["time"], np.full(1000, 128, dtype=np.uint16)),
                "ICIE__RAD_SAMPLE_1": (["time"], np.random.rand(1000)),
            }
        )

        nom_hk_data = xr.Dataset(
            {
                "PACKET_ICIE_TIME": (["time"], pd.date_range("2025-01-01", periods=100, freq="10ms").values),
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
            patch("libera_rad.l1b.KernelManager"),
            patch("libera_rad.radiometer.radiance.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
            patch("libera_rad.geolocation.calculate_geolocation_for_timestamps") as mock_geoloc,
            patch("libera_rad.geolocation.calculate_geometry") as mock_geometry,
            patch("libera_rad.geolocation.add_moon_boresight_offsets", side_effect=lambda _km, df: df),
            patch("libera_rad.geolocation.calculate_start_of_hour_state") as mock_hourly,
            patch("libera_rad.geolocation.calculate_azimuth_elevation_for_timestamps") as mock_azel,
            patch("libera_rad.geolocation.moon_in_field_of_view") as mock_moon_fov,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_radiance,
        ):
            # Setup mocks
            mock_cal = Mock()
            channel_prop = Mock()
            channel_prop.channel_enum = "1"
            mock_cal.channels = {"sw": channel_prop}
            mock_load_cal.return_value = mock_cal

            mock_downsample.side_effect = lambda x, **kwargs: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)
            instrument_lla = pd.DataFrame(
                {
                    "lat": np.linspace(30, 31, 100, dtype=np.float32),
                    "lon": np.linspace(-100, -99, 100, dtype=np.float32),
                    "alt": np.zeros(100, dtype=np.float32),
                }
            )
            mock_geoloc.return_value = instrument_lla
            subsat_lat = np.linspace(29, 30, 100, dtype=np.float32)
            mock_geometry.return_value = pd.DataFrame(
                {
                    "subsatellite_latitude": subsat_lat,
                    "subsatellite_longitude": np.linspace(-101, -100, 100, dtype=np.float32),
                    "subsatellite_colatitude": (90.0 - subsat_lat).astype(np.float32),
                    "subsolar_latitude": np.linspace(-23, -22, 100, dtype=np.float32),
                    "subsolar_longitude": np.linspace(40, 41, 100, dtype=np.float32),
                    "subsolar_colatitude": np.linspace(113, 112, 100, dtype=np.float32),
                    "spacecraft_radius": np.full(100, 7000.0, dtype=np.float64),
                    "spacecraft_altitude": np.zeros(100, dtype=np.float32),
                    "spacecraft_position_inertial_x": np.full(100, 5000.0, dtype=np.float64),
                    "spacecraft_position_inertial_y": np.full(100, 4000.0, dtype=np.float64),
                    "spacecraft_position_inertial_z": np.full(100, 3000.0, dtype=np.float64),
                    "boresight_inertial_x": np.full(100, -1.0, dtype=np.float64),
                    "boresight_inertial_y": np.zeros(100, dtype=np.float64),
                    "boresight_inertial_z": np.zeros(100, dtype=np.float64),
                    "viewing_zenith": np.linspace(10, 20, 100, dtype=np.float64),
                    "solar_zenith": np.linspace(40, 50, 100, dtype=np.float64),
                    "viewing_azimuth": np.linspace(100, 110, 100, dtype=np.float64),
                    "solar_azimuth": np.linspace(120, 130, 100, dtype=np.float64),
                    "relative_azimuth": np.linspace(150, 160, 100, dtype=np.float64),
                    "cone_angle": np.linspace(5, 15, 100, dtype=np.float64),
                    "cone_angle_rate": np.linspace(-1, 1, 100, dtype=np.float64),
                    "spacecraft_velocity_inertial_x": np.full(100, -2.9, dtype=np.float64),
                    "spacecraft_velocity_inertial_y": np.full(100, 3.5, dtype=np.float64),
                    "spacecraft_velocity_inertial_z": np.full(100, 6.0, dtype=np.float64),
                    "attitude_q0": np.full(100, 1.0, dtype=np.float64),
                    "attitude_q1": np.zeros(100, dtype=np.float64),
                    "attitude_q2": np.zeros(100, dtype=np.float64),
                    "attitude_q3": np.zeros(100, dtype=np.float64),
                    "clock_angle": np.linspace(90, 270, 100, dtype=np.float64),
                    "clock_angle_rate": np.linspace(-5, 5, 100, dtype=np.float64),
                    "along_track_angle": np.full(100, -0.16, dtype=np.float64),
                    "cross_track_angle": np.linspace(-60, 60, 100, dtype=np.float64),
                    "moon_boresight_angle": np.linspace(2, 40, 100, dtype=np.float64),
                    "moon_azimuth_offset": np.linspace(-30, 30, 100, dtype=np.float64),
                    "moon_elevation_offset": np.linspace(5, -5, 100, dtype=np.float64),
                    "moon_angular_radius": np.full(100, 0.26, dtype=np.float64),
                    "moon_distance": np.full(100, 3.8e5, dtype=np.float64),
                }
            )
            mock_hourly.return_value = (np.full((24, 3), 7000.0), np.full((24, 3), 7.5))
            moon_flags = (np.linspace(2, 40, 100) <= 6.26).astype(np.int8)
            mock_moon_fov.return_value = moon_flags
            mock_azel.return_value = (np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32))
            mock_radiance.return_value = pd.Series(np.random.rand(100))

            result, dynamic_attributes = l1b.process_l1a_to_l1b(mock_input_data, dynamic_kernel_sources, use_geo=True)

            mock_geoloc.assert_called_once()
            mock_azel.assert_called_once()

            # Check result structure
            assert isinstance(result, dict)
            assert "radiometer_time" in result
            assert "Latitude" in result
            assert "Filtered_Radiance_SW" in result
            assert isinstance(dynamic_attributes, dict)
            assert "Earth_Sun_Distance_AU" in dynamic_attributes
            assert np.all(result["Subsatellite_Latitude"] != np.float32(-999))
            assert np.allclose(result["Colatitude"], 90.0 - result["Latitude"])
            assert np.allclose(result["Subsatellite_Colatitude"], 90.0 - result["Subsatellite_Latitude"])
            assert not np.any(result["Subsolar_Longitude"] == np.float32(-999))
            assert np.allclose(result["Radius_of_Satellite_from_Center_of_Earth"], 7000.0, rtol=0.0, atol=1e-6)
            assert np.all(result["Azimuth"] == np.float32(0))
            assert np.all(result["Elevation"] == np.float32(0))
            # Boresight surface geometry is populated from curryer. Assert the mocked values
            # themselves: a not-equal-to-fill check would also pass on an all-NaN array.
            assert np.allclose(result["Solar_Zenith_Surface"], np.linspace(40, 50, 100), atol=1e-4)
            assert np.allclose(result["Viewing_Zenith_Surface"], np.linspace(10, 20, 100), atol=1e-4)
            assert np.allclose(result["Viewing_Azimuth_Surface_WRT_North"], np.linspace(100, 110, 100), atol=1e-4)
            assert np.allclose(result["Relative_Azimuth_Surface"], np.linspace(150, 160, 100), atol=1e-4)
            assert np.allclose(result["Cone_Angle"], np.linspace(5, 15, 100), atol=1e-4)
            assert np.allclose(result["Cone_Angle_Rate"], np.linspace(-1, 1, 100), atol=1e-4)
            assert result["Satellite_Position"].shape == (len(result["Latitude"]), 3)
            assert result["Line_Of_Sight"].shape == (len(result["Latitude"]), 3)
            assert not np.any(result["Satellite_Position"] == np.float64(-9999))
            # Motion / attitude / orbital-frame fields from curryer.
            assert result["Satellite_Velocity"].shape == (len(result["Latitude"]), 3)
            assert np.allclose(result["Satellite_Attitude_Q0"], 1.0)
            assert not np.any(result["Clock_Angle"] == np.float32(-999))
            assert np.allclose(result["Along_Track_Angle"], -0.16, atol=1e-4)
            assert np.allclose(result["Cross_Track_Angle"], np.linspace(-60, 60, 100), atol=1e-3)
            # Clock rate is filled inside the nadir cone gate (mocked cone spans 5..15 deg).
            inside_cone = np.linspace(5, 15, 100) < float(CLOCK_RATE_MIN_CONE_ANGLE_DEG)
            # Guard: the mocked cone angles must straddle the gate so both sides are exercised.
            assert inside_cone.any()
            assert not inside_cone.all()
            assert np.all(result["Clock_Angle_Rate"][inside_cone] == np.float32(-999))
            assert not np.any(result["Clock_Angle_Rate"][~inside_cone] == np.float32(-999))
            # Start-of-hour state on the fixed 24-hour grid.
            assert result["Satellite_Position_Start_Of_Hour"].shape == (24, 3)
            assert result["Satellite_Velocity_Start_Of_Hour"].shape == (24, 3)
            assert np.allclose(result["Satellite_Position_Start_Of_Hour"], 7000.0)
            assert np.allclose(result["Satellite_Velocity_Start_Of_Hour"], 7.5)
            # Lunar geometry: the offsets pass through curryer unchanged and the flag is
            # the predicate's, evaluated inside the kernel context alongside the geometry.
            mock_moon_fov.assert_called_once()
            assert np.allclose(result["Moon_Boresight_Angle"], np.linspace(2, 40, 100), atol=1e-3)
            assert np.allclose(result["Moon_Azimuth_Offset"], np.linspace(-30, 30, 100), atol=1e-3)
            assert np.allclose(result["Moon_Elevation_Offset"], np.linspace(5, -5, 100), atol=1e-3)
            np.testing.assert_array_equal(result["Moon_In_Field_Of_View"], moon_flags)
            assert result["Moon_In_Field_Of_View"].dtype == np.int8
            # Guard: the mocked sweep must cross the threshold so both states are exercised.
            assert moon_flags.any()
            assert not moon_flags.all()

    def test_process_l1a_to_l1b_use_geo_false(self, mock_input_data):
        """use_geo false should bypass KernelManager and SPICE geolocation."""
        with (
            patch("libera_rad.radiometer.radiance._load_calibration_data") as mock_load_cal,
            patch("libera_rad.l1b.KernelManager") as mock_kernel_manager_cls,
            patch("libera_rad.radiometer.radiance.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
            patch("libera_rad.geolocation.create_placeholder_geolocation_dataframe") as mock_placeholder_geo,
            patch("libera_rad.geolocation.create_placeholder_azimuth_elevation") as mock_placeholder_azel,
            patch("libera_rad.geolocation.calculate_geolocation_for_timestamps") as mock_calculate_geo,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_radiance,
        ):
            mock_cal = Mock()
            channel_prop = Mock()
            channel_prop.channel_enum = "1"
            mock_cal.channels = {"sw": channel_prop}
            mock_load_cal.return_value = mock_cal

            mock_downsample.side_effect = lambda x, **kwargs: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)
            mock_placeholder_geo.return_value = pd.DataFrame(
                {
                    "lat": np.full(100, -999, dtype=np.float32),
                    "lon": np.full(100, -999, dtype=np.float32),
                    "alt": np.full(100, -9999, dtype=np.float32),
                }
            )
            mock_placeholder_azel.return_value = (
                np.full(100, -999, dtype=np.float32),
                np.full(100, -999, dtype=np.float32),
            )
            mock_radiance.return_value = pd.Series(np.random.rand(100))

            result, dynamic_attributes = l1b.process_l1a_to_l1b(mock_input_data, [], use_geo=False)

        mock_kernel_manager_cls.assert_not_called()
        mock_calculate_geo.assert_not_called()
        mock_placeholder_geo.assert_called_once_with(100)
        mock_placeholder_azel.assert_called_once_with(100)
        assert isinstance(result, dict)
        assert "Latitude" in result
        assert np.all(result["Latitude"] == np.float32(-999))
        assert np.all(result["Colatitude"] == np.float32(-999))
        assert np.all(result["Subsatellite_Colatitude"] == np.float32(-999))
        assert np.all(result["Azimuth"] == np.float32(-999))
        assert np.all(result["Elevation"] == np.float32(-999))
        # Geolocation is bypassed, so the Moon's position is unknown: the offsets take the
        # angle fill and the flag its own fill, never 0 (which would assert the Moon was out).
        assert np.all(result["Moon_Boresight_Angle"] == np.float32(-999))
        assert np.all(result["Moon_Azimuth_Offset"] == np.float32(-999))
        assert np.all(result["Moon_Elevation_Offset"] == np.float32(-999))
        assert np.all(result["Moon_In_Field_Of_View"] == MOON_IN_FOV_FILL_VALUE)
        assert isinstance(dynamic_attributes, dict)

    def test_process_l1a_to_l1b_jpss_only_mode(self, mock_input_data):
        """jpss_only uses LIBERA_BASE geo and zero motor angles."""
        dynamic_kernel_sources = [
            "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bsp",
            "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26092192956.bc",
        ]
        with (
            patch("libera_rad.radiometer.radiance._load_calibration_data") as mock_load_cal,
            patch("libera_rad.l1b.KernelManager"),
            patch("libera_rad.radiometer.radiance.downsample_libera_signal") as mock_downsample,
            patch("libera_rad.radiometer.gain_calibration.apply_gain_calibration") as mock_calibrate,
            patch("libera_rad.radiometer.gain_calibration.get_ground_cal_response_function") as mock_response,
            patch("libera_rad.geolocation.calculate_geometry") as mock_geometry,
            patch("libera_rad.geolocation.add_moon_boresight_offsets", side_effect=lambda _km, df: df),
            patch("libera_rad.geolocation.calculate_start_of_hour_state") as mock_hourly,
            patch("libera_rad.geolocation.calculate_geolocation_for_timestamps") as mock_prod_geo,
            patch("libera_rad.geolocation.moon_in_field_of_view") as mock_moon_fov,
            patch("libera_rad.radiometer.radiance.calculate_radiance") as mock_radiance,
        ):
            mock_cal = Mock()
            channel_prop = Mock()
            channel_prop.channel_enum = "1"
            mock_cal.channels = {"sw": channel_prop}
            mock_load_cal.return_value = mock_cal
            mock_downsample.side_effect = lambda x, **kwargs: x[::10]
            mock_calibrate.return_value = np.random.rand(1000)
            mock_response.return_value = np.ones(501)
            subsat_lat = np.linspace(10, 11, 100, dtype=np.float32)
            mock_geometry.return_value = pd.DataFrame(
                {
                    "subsatellite_latitude": subsat_lat,
                    "subsatellite_longitude": np.linspace(-80, -79, 100, dtype=np.float32),
                    "subsatellite_colatitude": (90.0 - subsat_lat).astype(np.float32),
                    "subsolar_latitude": np.linspace(-23, -22, 100, dtype=np.float32),
                    "subsolar_longitude": np.linspace(40, 41, 100, dtype=np.float32),
                    "subsolar_colatitude": np.linspace(113, 112, 100, dtype=np.float32),
                    "spacecraft_radius": np.full(100, 7000.0, dtype=np.float64),
                    "spacecraft_altitude": np.zeros(100, dtype=np.float32),
                    "spacecraft_position_inertial_x": np.full(100, 5000.0, dtype=np.float64),
                    "spacecraft_position_inertial_y": np.full(100, 4000.0, dtype=np.float64),
                    "spacecraft_position_inertial_z": np.full(100, 3000.0, dtype=np.float64),
                    # No motor CK in jpss_only, so the boresight-derived fields are NaN.
                    "boresight_inertial_x": np.full(100, np.nan),
                    "boresight_inertial_y": np.full(100, np.nan),
                    "boresight_inertial_z": np.full(100, np.nan),
                    "viewing_zenith": np.full(100, np.nan),
                    "solar_zenith": np.full(100, np.nan),
                    "viewing_azimuth": np.full(100, np.nan),
                    "solar_azimuth": np.full(100, np.nan),
                    "relative_azimuth": np.full(100, np.nan),
                    "cone_angle": np.full(100, np.nan),
                    "cone_angle_rate": np.full(100, np.nan),
                    "clock_angle": np.full(100, np.nan),
                    "clock_angle_rate": np.full(100, np.nan),
                    "along_track_angle": np.full(100, np.nan),
                    "cross_track_angle": np.full(100, np.nan),
                    "moon_boresight_angle": np.full(100, np.nan),
                    "moon_azimuth_offset": np.full(100, np.nan),
                    "moon_elevation_offset": np.full(100, np.nan),
                    "moon_angular_radius": np.full(100, np.nan),
                    "moon_distance": np.full(100, np.nan),
                    # Velocity and body attitude come from the spacecraft observer, so they
                    # resolve in jpss_only (SPK + JPSS CK) even without the motor CK.
                    "spacecraft_velocity_inertial_x": np.full(100, -2.9, dtype=np.float64),
                    "spacecraft_velocity_inertial_y": np.full(100, 3.5, dtype=np.float64),
                    "spacecraft_velocity_inertial_z": np.full(100, 6.0, dtype=np.float64),
                    "attitude_q0": np.full(100, 1.0, dtype=np.float64),
                    "attitude_q1": np.zeros(100, dtype=np.float64),
                    "attitude_q2": np.zeros(100, dtype=np.float64),
                    "attitude_q3": np.zeros(100, dtype=np.float64),
                }
            )
            mock_hourly.return_value = (np.full((24, 3), 7000.0), np.full((24, 3), 7.5))
            mock_moon_fov.return_value = np.full(100, MOON_IN_FOV_FILL_VALUE, dtype=np.int8)
            mock_radiance.return_value = pd.Series(np.random.rand(100))

            result, _ = l1b.process_l1a_to_l1b(mock_input_data, dynamic_kernel_sources, jpss_only_mode=True)

        mock_prod_geo.assert_not_called()
        mock_geometry.assert_called_once()
        assert np.all(result["Azimuth"] == 0)
        assert np.all(result["Elevation"] == 0)
        assert np.allclose(result["Subsatellite_Latitude"], result["Latitude"])
        assert np.allclose(result["Subsatellite_Longitude"], result["Longitude"])
        assert np.allclose(result["Colatitude"], 90.0 - result["Latitude"])
        assert np.allclose(result["Subsatellite_Colatitude"], 90.0 - result["Subsatellite_Latitude"])
        # Subsolar point and satellite radius are populated from curryer.
        assert not np.any(result["Subsolar_Latitude"] == np.float32(-999))
        assert np.allclose(result["Radius_of_Satellite_from_Center_of_Earth"], 7000.0, rtol=0.0, atol=1e-6)
        # Every boresight-derived field is NaN in jpss_only (no motor CK), not just some.
        for name in (
            "Solar_Zenith_Surface",
            "Viewing_Zenith_Surface",
            "Viewing_Azimuth_Surface_WRT_North",
            "Solar_Azimuth_Surface_WRT_North",
            "Relative_Azimuth_Surface",
            "Cone_Angle",
            "Cone_Angle_Rate",
            "Clock_Angle",
            "Clock_Angle_Rate",
            "Along_Track_Angle",
            "Cross_Track_Angle",
            "Moon_Boresight_Angle",
            "Moon_Azimuth_Offset",
            "Moon_Elevation_Offset",
        ):
            assert np.all(np.isnan(result[name])), f"{name} should be NaN in jpss_only mode"
        # With no boresight there is no Moon-in-view answer, so the flag fills rather
        # than claiming the Moon was absent.
        mock_moon_fov.assert_called_once()
        assert np.all(result["Moon_In_Field_Of_View"] == MOON_IN_FOV_FILL_VALUE)
        # Spacecraft-observer fields still resolve without the motor CK.
        assert not np.any(result["Satellite_Position"] == np.float64(-9999))
        assert not np.any(result["Satellite_Velocity"] == np.float64(-999))
        assert np.allclose(result["Satellite_Attitude_Q0"], 1.0)
        assert np.allclose(result["Satellite_Position_Start_Of_Hour"], 7000.0)

    @pytest.mark.parametrize("dynamic_kernel_sources", [None, []])
    def test_process_l1a_to_l1b_requires_kernel_sources_when_use_geo_true(
        self, mock_input_data, dynamic_kernel_sources
    ):
        """Geolocation mode requires SPICE kernel paths from the manifest."""
        timestamps = np.arange(10, dtype=np.float64)
        calibrated_data = {"sw": np.zeros(10)}

        with patch(
            "libera_rad.radiometer.radiance.calibrate_and_downsample_radiometer_data",
            return_value=(timestamps, calibrated_data),
        ):
            with pytest.raises(
                ValueError,
                match="SPICE kernel sources are required for geolocation when use_geo is True",
            ):
                l1b.process_l1a_to_l1b(mock_input_data, dynamic_kernel_sources, use_geo=True)


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

    def test_algorithm_rejects_use_geo_false_and_jpss_only(self, tmp_path, monkeypatch):
        """Mutually exclusive manifest flags raise before processing."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = tmp_path / "manifest.json"
        with (
            patch("libera_utils.Manifest.from_file") as mock_from_file,
            patch("libera_rad.l1b.read_all_input_data"),
        ):
            mock_from_file.return_value = Mock(
                files=[manifest_path],
                configuration={"use_geo": False, "jpss_only": True},
            )
            with pytest.raises(ValueError, match="cannot both be enabled"):
                l1b.algorithm(manifest_path)

    def test_algorithm_use_geo_false_disables_geolocation(self, tmp_path, monkeypatch):
        """Explicit use_geo: false disables SPICE geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = tmp_path / "manifest.json"
        output_manifest = Mock()
        with (
            patch("libera_utils.Manifest.from_file") as mock_from_file,
            patch("libera_utils.Manifest.output_manifest_from_input_manifest", return_value=output_manifest),
            patch("libera_rad.l1b.read_all_input_data") as mock_read,
            patch("libera_rad.l1b.process_l1a_to_l1b") as mock_process,
            patch("libera_rad.l1b.create_and_write_data_product") as mock_write,
        ):
            mock_from_file.return_value = Mock(
                files=[manifest_path],
                configuration={"use_geo": False},
            )
            mock_read.return_value = ({}, [])
            mock_process.return_value = ({}, {})
            mock_write.return_value = Mock(path="out.nc")
            output_manifest.write.return_value = tmp_path / "out_manifest.json"
            input_manifest = mock_from_file.return_value
            l1b.algorithm(manifest_path)
            mock_read.assert_called_once_with(input_manifest)
            assert mock_process.call_args.kwargs["use_geo"] is False
            assert mock_process.call_args.kwargs["jpss_only_mode"] is False

    def test_algorithm_omitted_use_geo_defaults_to_true(self, tmp_path, monkeypatch):
        """Omitting use_geo from configuration defaults to production geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = tmp_path / "manifest.json"
        output_manifest = Mock()
        with (
            patch("libera_utils.Manifest.from_file") as mock_from_file,
            patch("libera_utils.Manifest.output_manifest_from_input_manifest", return_value=output_manifest),
            patch("libera_rad.l1b.read_all_input_data") as mock_read,
            patch("libera_rad.l1b.process_l1a_to_l1b") as mock_process,
            patch("libera_rad.l1b.create_and_write_data_product") as mock_write,
        ):
            mock_from_file.return_value = Mock(files=[manifest_path], configuration={})
            mock_read.return_value = ({}, [])
            mock_process.return_value = ({}, {})
            mock_write.return_value = Mock(path="out.nc")
            output_manifest.write.return_value = tmp_path / "out_manifest.json"
            input_manifest = mock_from_file.return_value
            l1b.algorithm(manifest_path)
            mock_read.assert_called_once_with(input_manifest)
            assert mock_process.call_args.kwargs["use_geo"] is True
            assert mock_process.call_args.kwargs["jpss_only_mode"] is False

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
        algo_inputs = generate_input_manifest()

        # Run the algorithm
        output_manifest_path = l1b.algorithm(algo_inputs)

        output_manifest_obj = Manifest.from_file(output_manifest_path)

        for file in output_manifest_obj.files:
            data_product = xr.open_dataset(file.filename)
            print(data_product)
