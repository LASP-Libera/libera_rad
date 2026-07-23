"""Integration tests for L1B manifest configuration.use_geo."""

import numpy as np
import pytest
import spiceypy as sp
import xarray as xr
from libera_utils.io.manifest import Manifest
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad import l1b
from libera_rad.geolocation import calculate_geolocation_for_timestamps, calculate_geometry
from libera_rad.radiometer.radiance import calibrate_and_downsample_radiometer_data

_LATITUDE_FILL = np.float32(-999)
_LONGITUDE_FILL = np.float32(-999)
_ALTITUDE_FILL = np.float32(-9999)

_INTEGRATION_KERNEL_FILENAMES = (
    "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26016220328.bc",
    "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26016220138.bc",
    "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp",
    "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc",
)
_RAD_SAMPLE_FILENAME = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"


@pytest.fixture(scope="module")
def expected_geolocation(test_integration_data_path):
    """Direct SPICE geolocation for the L1B integration radiometer timestamps."""
    rad_data = xr.open_dataset(test_integration_data_path / _RAD_SAMPLE_FILENAME).load()
    timestamps, _ = calibrate_and_downsample_radiometer_data(rad_data)
    kernel_sources = [str(test_integration_data_path / name) for name in _INTEGRATION_KERNEL_FILENAMES]

    with KernelManager() as km:
        km.load_libera_dynamic_kernels(kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)
        instrument_lla = calculate_geolocation_for_timestamps(km, timestamps)
        return instrument_lla


class TestL1bManifestUseGeoConfiguration:
    """Each manifest configuration case is a separate integration test."""

    @pytest.fixture(autouse=True)
    def clear_spice_state(self):
        try:
            sp.kclear()
        except Exception:
            pass
        yield
        try:
            sp.kclear()
        except Exception:
            pass

    def test_use_geo_absent_runs_spice_geolocation(
        self, generate_input_manifest, monkeypatch, tmp_path, expected_geolocation
    ):
        """Omitting use_geo from configuration runs SPICE geolocation (production default)."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = generate_input_manifest({})
        input_manifest = Manifest.from_file(manifest_path)

        output_manifest = Manifest.from_file(l1b.algorithm(manifest_path))
        for key, value in input_manifest.configuration.items():
            assert output_manifest.configuration[key] == value

        with xr.open_dataset(output_manifest.files[0].filename, mask_and_scale=False) as dataset:
            lat = dataset["Latitude"].values
            lon = dataset["Longitude"].values
            alt = dataset["Altitude"].values

            assert np.sum(np.isfinite(lat)) > 1000, "Expected many finite SPICE latitude values"
            assert np.all((lat[np.isfinite(lat)] >= -90) & (lat[np.isfinite(lat)] <= 90))
            assert np.all((lon[np.isfinite(lon)] >= -180) & (lon[np.isfinite(lon)] <= 180))

            np.testing.assert_allclose(lat, expected_geolocation["lat"].to_numpy().astype(np.float32), equal_nan=True)
            np.testing.assert_allclose(lon, expected_geolocation["lon"].to_numpy().astype(np.float32), equal_nan=True)
            np.testing.assert_allclose(alt, expected_geolocation["alt"].to_numpy().astype(np.float32), equal_nan=True)

    def test_use_geo_true_runs_spice_geolocation(
        self, generate_input_manifest, monkeypatch, tmp_path, expected_geolocation
    ):
        """Explicit use_geo true runs SPICE geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = generate_input_manifest({"use_geo": True})
        input_manifest = Manifest.from_file(manifest_path)

        output_manifest = Manifest.from_file(l1b.algorithm(manifest_path))
        for key, value in input_manifest.configuration.items():
            assert output_manifest.configuration[key] == value

        with xr.open_dataset(output_manifest.files[0].filename, mask_and_scale=False) as dataset:
            lat = dataset["Latitude"].values
            lon = dataset["Longitude"].values
            alt = dataset["Altitude"].values

            assert np.sum(np.isfinite(lat)) > 1000, "Expected many finite SPICE latitude values"
            np.testing.assert_allclose(lat, expected_geolocation["lat"].to_numpy().astype(np.float32), equal_nan=True)
            np.testing.assert_allclose(lon, expected_geolocation["lon"].to_numpy().astype(np.float32), equal_nan=True)
            np.testing.assert_allclose(alt, expected_geolocation["alt"].to_numpy().astype(np.float32), equal_nan=True)

    def test_use_geo_false_writes_placeholder_geolocation(self, generate_input_manifest, monkeypatch, tmp_path):
        """Explicit use_geo false skips SPICE and writes standard lat/lon/alt fill values."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        manifest_path = generate_input_manifest({"use_geo": False})
        input_manifest = Manifest.from_file(manifest_path)

        output_manifest = Manifest.from_file(l1b.algorithm(manifest_path))
        for key, value in input_manifest.configuration.items():
            assert output_manifest.configuration[key] == value

        with xr.open_dataset(output_manifest.files[0].filename, mask_and_scale=False) as dataset:
            assert np.all(dataset["Latitude"].values == _LATITUDE_FILL)
            assert np.all(dataset["Longitude"].values == _LONGITUDE_FILL)
            assert np.all(dataset["Altitude"].values == _ALTITUDE_FILL)
            assert np.all(dataset["Azimuth"].values == _LATITUDE_FILL)
            assert np.all(dataset["Elevation"].values == _LATITUDE_FILL)
            assert np.any(dataset["Filtered_Radiance_SW"].values != np.float32(-999))

        with xr.open_dataset(output_manifest.files[0].filename) as dataset:
            assert not np.any(np.isfinite(dataset["Latitude"].values))
            assert not np.any(np.isfinite(dataset["Longitude"].values))
            assert not np.any(np.isfinite(dataset["Altitude"].values))


class TestGeometryErrorHandling:
    """SPICE failures from the curryer geometry query surface as parsed, user-facing errors."""

    @pytest.fixture(autouse=True)
    def clear_spice_state(self):
        try:
            sp.kclear()
        except Exception:
            pass
        yield
        try:
            sp.kclear()
        except Exception:
            pass

    def test_out_of_coverage_request_raises_friendly_error(self, test_integration_data_path):
        """A geometry request outside kernel coverage must raise a parsed RuntimeError, not a raw SpiceyError dump."""
        kernel_sources = [str(test_integration_data_path / name) for name in _INTEGRATION_KERNEL_FILENAMES]
        # The loaded kernels cover 2025-11-20; request timestamps years outside that window.
        out_of_coverage = np.array(["2000-01-01T00:00:00", "2000-01-01T00:00:01"], dtype="datetime64[ns]")
        with KernelManager() as km:
            km.load_libera_dynamic_kernels(kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)
            with pytest.raises(RuntimeError) as excinfo:
                calculate_geometry(km, out_of_coverage)

        message = str(excinfo.value)
        assert "JPSS4_SC" in message, message
        assert "coverage" in message.lower(), message
        # A parsed cause, not the raw multi-line NAIF traceback dump.
        assert "Toolkit version" not in message, message
