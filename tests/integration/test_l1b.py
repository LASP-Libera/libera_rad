"""Integration Tests for the L1B algorithm."""

import os
from datetime import UTC, date, datetime

import numpy as np
import pytest
import xarray as xr
import yaml
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType

from libera_rad import l1b
from libera_rad.calibration.constants import ChannelName
from libera_rad.config import product_config_path
from libera_rad.radiometer.radiance import (
    calculate_radiances,
    calibrate_and_downsample_radiometer_data,
    interpolate_temperatures,
)
from libera_rad.version import version as libera_rad_version

# Mapping from ChannelName enum values to L1B product variable names
_CHANNEL_TO_RADIANCE_VAR: dict[str, str] = {
    ChannelName.SHORTWAVE.value: "Filtered_Radiance_SW",
    ChannelName.LONGWAVE.value: "Filtered_Radiance_LW",
    ChannelName.TOTAL.value: "Filtered_Radiance_Tot",
    ChannelName.SPLIT_SHORTWAVE.value: "Filtered_Radiance_SSW",
}


@pytest.fixture(scope="module")
def l1b_product_file_path(tmp_path_factory, test_integration_data_path):
    """Run the L1B algorithm once per test module and return the output data file path."""
    libera_files = [
        test_integration_data_path
        / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc",
        test_integration_data_path / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc",
        test_integration_data_path / "LIBERA_SPICE_ELSCAN-CK_V5-5-1_20251120T175950_20251120T190549_R26016220328.bc",
        test_integration_data_path / "LIBERA_SPICE_AZROT-CK_V5-5-1_20251120T175950_20251120T190549_R26016220138.bc",
        test_integration_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp",
        test_integration_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc",
    ]
    tmp_path = tmp_path_factory.mktemp("l1b_product")
    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=libera_files, configuration={})

    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )
    manifest_path = str(input_manifest.write(out_path=tmp_path))

    previous_processing_path = os.environ.get("PROCESSING_PATH")
    os.environ["PROCESSING_PATH"] = str(tmp_path)
    try:
        output_manifest_path = l1b.algorithm(manifest_path)
    finally:
        if previous_processing_path is None:
            os.environ.pop("PROCESSING_PATH", None)
        else:
            os.environ["PROCESSING_PATH"] = previous_processing_path

    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1, "Expected exactly one L1B output file in manifest"
    return output_manifest.files[0].filename


@pytest.fixture(scope="module")
def l1b_product_dataset(l1b_product_file_path):
    """Open the L1B output with default decoding for science and invariant checks."""
    with xr.open_dataset(l1b_product_file_path) as ds:
        return ds.load()


@pytest.fixture(scope="module")
def l1b_product_dataset_raw(l1b_product_file_path):
    """Open the L1B output without mask/scale decoding for strict schema dtype checks."""
    with xr.open_dataset(l1b_product_file_path, mask_and_scale=False) as ds:
        return ds.load()


class TestL1bManifest:
    """Tests that the output manifest has correct structure and naming."""

    def test_l1b_manifest(self, generate_input_manifest, monkeypatch, tmp_path):
        """Output manifest must have type OUTPUT, exactly one file, and a valid Libera filename."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest_path = l1b.algorithm(generate_input_manifest())

        actual = Manifest.from_file(output_manifest_path)
        # Checksums, UIDs, and full paths will differ, but format and file names must be valid
        assert len(actual.files) == 1
        assert actual.manifest_type == "OUTPUT"
        assert LiberaDataProductFilename.from_file_path(output_manifest_path)


class TestL1bProductDefinitionCompliance:
    """Tests that the L1B output conforms to the product definition schema. This is implicitly tested by just making
    a data product successfully. These tests are more explicit for clarity and finer-grained failure diagnostics.

    Assertions are driven by libera_rad/data/L1B_RAD-4CH_product_definition.yml so
    they automatically adapt when the product definition is updated.
    """

    @pytest.fixture(scope="class")
    def product_definition(self):
        """Load and return the product definition YAML as a dict."""
        return yaml.safe_load(product_config_path.read_text())

    def test_all_defined_variables_present(self, l1b_product_dataset_raw, product_definition):
        """All variables declared in the product definition must appear in the output."""
        assert set(l1b_product_dataset_raw.data_vars) == set(product_definition["variables"])

    def test_all_defined_coordinates_present(self, l1b_product_dataset_raw, product_definition):
        """All coordinates declared in the product definition must appear in the output."""
        assert set(product_definition["coordinates"]).issubset(set(l1b_product_dataset_raw.coords))

    def test_variable_dtypes_match_definition(self, l1b_product_dataset_raw, product_definition):
        """Each variable's dtype must match the product definition."""
        for var_name, var_def in product_definition["variables"].items():
            expected_dtype = np.dtype(var_def["dtype"])
            actual_dtype = l1b_product_dataset_raw[var_name].dtype
            assert actual_dtype == expected_dtype, f"{var_name}: expected dtype {expected_dtype}, got {actual_dtype}"

    def test_variable_dimensions_match_definition(self, l1b_product_dataset_raw, product_definition):
        """Each variable's dimension names must match the product definition."""
        for var_name, var_def in product_definition["variables"].items():
            expected_dims = tuple(var_def["dimensions"])
            actual_dims = l1b_product_dataset_raw[var_name].dims
            assert actual_dims == expected_dims, f"{var_name}: expected dims {expected_dims}, got {actual_dims}"

    def test_required_global_attributes_present(self, l1b_product_dataset, product_definition):
        """All top-level attribute keys in the product definition must be present in the output."""
        for attr_name in product_definition.get("attributes", {}):
            assert attr_name in l1b_product_dataset.attrs, (
                f"Required global attribute '{attr_name}' not found in output dataset"
            )

    def test_algorithm_version_matches_package(self, l1b_product_dataset):
        """The algorithm_version global attribute must match the installed package version."""
        assert l1b_product_dataset.attrs["algorithm_version"] == libera_rad_version()


class TestL1bScienceValues:
    """Tests that implemented pipeline fields contain real computed values, not fill sentinels."""

    @pytest.fixture(scope="class")
    def product_definition(self):
        """Load and return the product definition YAML as a dict."""
        return yaml.safe_load(product_config_path.read_text())

    def test_radiometer_time_monotonically_increasing(self, l1b_product_dataset):
        """Timestamps must be strictly increasing across all radiometer samples."""
        times = l1b_product_dataset["radiometer_time"].values.astype(np.int64)
        assert np.all(np.diff(times) > 0), "radiometer_time is not strictly monotonically increasing"

    @pytest.mark.parametrize(
        "var_name",
        [
            "Filtered_Radiance_SW",
            "Filtered_Radiance_LW",
            "Filtered_Radiance_Tot",
            "Filtered_Radiance_SSW",
        ],
    )
    def test_radiance_channels_contain_computed_values(self, l1b_product_dataset, product_definition, var_name):
        """Each radiance variable must have at least one value that is not the fill sentinel."""
        fill_value = product_definition["variables"][var_name]["attributes"]["_FillValue"]
        vals = l1b_product_dataset[var_name].values
        assert np.any(vals != fill_value), f"{var_name} contains only fill-value sentinels"

    def test_latitude_longitude_contain_computed_values(self, l1b_product_dataset, product_definition):
        """Latitude and Longitude must each have at least one computed (non-fill) value."""
        for var_name in ("Latitude", "Longitude"):
            fill_value = product_definition["variables"][var_name]["attributes"]["_FillValue"]
            vals = l1b_product_dataset[var_name].values
            assert np.any(vals != fill_value), f"{var_name} contains only fill-value sentinels"

    @pytest.mark.parametrize("var_name", ["Azimuth", "Elevation"])
    def test_azimuth_elevation_contain_computed_values(self, l1b_product_dataset, product_definition, var_name):
        """Azimuth/Elevation must have at least one computed (non-fill) value."""
        fill_value = product_definition["variables"][var_name]["attributes"]["_FillValue"]
        vals = l1b_product_dataset[var_name].values
        assert np.any(vals != fill_value), f"{var_name} contains only fill-value sentinels"

    def test_operational_mode_matches_fixture_obsid(self, l1b_product_dataset, product_definition):
        """Operational_Mode should reflect the fixture L1A OBSID (currently constant 128 in test data)."""
        fill_value = product_definition["variables"]["Operational_Mode"]["attributes"]["_FillValue"]
        vals = l1b_product_dataset["Operational_Mode"].values.astype(np.uint16)
        non_fill = vals[vals != np.uint16(fill_value)]
        assert len(non_fill) > 0, "Operational_Mode contains no non-fill samples"
        assert set(np.unique(non_fill)) == {np.uint16(128)}, "Unexpected Operational_Mode values for fixture"


class TestL1bRegressionStatistics:
    """Pinned summary statistics to detect unintended changes in derived fields."""

    @pytest.fixture(scope="class")
    def product_definition(self):
        """Load and return the product definition YAML as a dict."""
        return yaml.safe_load(product_config_path.read_text())

    # Expected values captured from the integration fixture output after implementation.
    # These numbers are intentionally tight to catch algorithm regressions;
    # update intentionally if kernels/config change.
    _AZIMUTH_MEAN_DEG = np.float64(359.9995557293844)
    _AZIMUTH_STD_DEG = np.float64(1.5156280262632314e-05)
    _ELEVATION_MEAN_DEG = np.float64(1.1046268158518922)
    _ELEVATION_STD_DEG = np.float64(44.820824)

    def test_azimuth_mean_std_pinned(self, l1b_product_dataset, product_definition):
        fill_value = product_definition["variables"]["Azimuth"]["attributes"]["_FillValue"]
        vals = l1b_product_dataset["Azimuth"].values.astype(np.float64)
        non_fill = vals[(vals != fill_value) & np.isfinite(vals)]
        assert len(non_fill) > 0, "Azimuth contains no finite, non-fill samples"
        mean = np.mean(non_fill)
        std = np.std(non_fill)
        assert np.isclose(mean, self._AZIMUTH_MEAN_DEG, rtol=1e-6, atol=1e-3)
        assert np.isclose(std, self._AZIMUTH_STD_DEG, rtol=1e-6, atol=1e-3)

    def test_elevation_mean_std_pinned(self, l1b_product_dataset, product_definition):
        fill_value = product_definition["variables"]["Elevation"]["attributes"]["_FillValue"]
        vals = l1b_product_dataset["Elevation"].values.astype(np.float64)
        non_fill = vals[(vals != fill_value) & np.isfinite(vals)]
        assert len(non_fill) > 0, "Elevation contains no finite, non-fill samples"
        mean = np.mean(non_fill)
        std = np.std(non_fill)
        assert np.isclose(mean, self._ELEVATION_MEAN_DEG, rtol=1e-6, atol=1e-3)
        assert np.isclose(std, self._ELEVATION_STD_DEG, rtol=1e-6, atol=1e-3)

    def test_radiance_values_match_pipeline_recompute(self, l1b_product_dataset, test_integration_data_path):
        """Radiance arrays in the product must agree with a direct call to the radiance sub-pipeline.

        This is the primary numeric regression guard: it validates pipeline correctness
        against the same L1A inputs without relying on any stored golden file.
        """
        rad_data = xr.open_dataset(
            test_integration_data_path
            / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"
        ).load()
        nom_hk_data = xr.open_dataset(
            test_integration_data_path
            / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc"
        ).load()

        timestamps, calibrated_data_by_channel = calibrate_and_downsample_radiometer_data(rad_data)
        interp_temps = interpolate_temperatures(timestamps, nom_hk_data)
        recomputed_radiances = calculate_radiances(calibrated_data_by_channel, interp_temps)

        for channel_value, var_name in _CHANNEL_TO_RADIANCE_VAR.items():
            expected = recomputed_radiances[channel_value].astype(np.float32)
            actual = l1b_product_dataset[var_name].values
            assert np.allclose(actual, expected, rtol=1e-5, equal_nan=True), (
                f"{var_name}: product values do not match direct pipeline recompute"
            )


class TestL1bPhysicalInvariants:
    """Broad physical plausibility checks independent of the product definition schema."""

    def test_variable_physicality(self, l1b_product_dataset):
        """Geolocation must contain finite samples; radiance channels must not contain NaN."""
        geolocation_vars = ["Latitude", "Longitude", "Altitude"]
        for var_name in geolocation_vars:
            vals = l1b_product_dataset[var_name].values
            assert np.any(np.isfinite(vals)), f"{var_name} has no finite geolocation values"

        radiance_vars = [
            "Filtered_Radiance_SW",
            "Filtered_Radiance_LW",
            "Filtered_Radiance_Tot",
            "Filtered_Radiance_SSW",
        ]
        for var_name in radiance_vars:
            vals = l1b_product_dataset[var_name].values
            assert not np.any(np.isnan(vals)), f"{var_name} contains unexpected NaN values"

    def test_non_fill_lat_lon_within_valid_range(self, l1b_product_dataset):
        """Non-fill Latitude values must be in [-90, 90] and Longitude values in [-180, 180]."""
        lat_fill, lon_fill = -999.0, -999.0
        lat = l1b_product_dataset["Latitude"].values
        lon = l1b_product_dataset["Longitude"].values
        non_fill_lat = lat[(lat != lat_fill) & np.isfinite(lat)]
        non_fill_lon = lon[(lon != lon_fill) & np.isfinite(lon)]

        assert len(non_fill_lat) > 0, "No finite, non-fill Latitude values available for range checks"
        assert len(non_fill_lon) > 0, "No finite, non-fill Longitude values available for range checks"

        assert np.all((non_fill_lat >= -90) & (non_fill_lat <= 90)), "Non-fill Latitude values are outside [-90, 90]"
        assert np.all((non_fill_lon >= -180) & (non_fill_lon <= 180)), (
            "Non-fill Longitude values are outside [-180, 180]"
        )

    def test_colatitude_populated_and_consistent_with_latitude(self, l1b_product_dataset):
        """Colatitude and Subsatellite_Colatitude must be 90 - latitude when geolocation is computed."""
        lat_fill = -999.0
        lat = l1b_product_dataset["Latitude"].values
        colat = l1b_product_dataset["Colatitude"].values
        subsat_lat = l1b_product_dataset["Subsatellite_Latitude"].values
        subsat_colat = l1b_product_dataset["Subsatellite_Colatitude"].values

        non_fill = (lat != lat_fill) & np.isfinite(lat)
        assert np.any(non_fill), "Expected computed instrument latitude values"
        assert np.allclose(colat[non_fill], 90.0 - lat[non_fill])
        assert np.allclose(subsat_colat[non_fill], 90.0 - subsat_lat[non_fill])
        assert np.all((colat[non_fill] >= 0) & (colat[non_fill] <= 180))
        assert np.all((subsat_colat[non_fill] >= 0) & (subsat_colat[non_fill] <= 180))

    def test_non_fill_azimuth_elevation_within_valid_range(self, l1b_product_dataset):
        """Non-fill Azimuth/Elevation values must be within the product definition valid ranges."""
        az_fill = -999.0
        el_fill = -999.0
        az = l1b_product_dataset["Azimuth"].values
        el = l1b_product_dataset["Elevation"].values

        az_nf = az[(az != az_fill) & np.isfinite(az)]
        el_nf = el[(el != el_fill) & np.isfinite(el)]

        assert len(az_nf) > 0, "No finite, non-fill Azimuth values available for range checks"
        assert len(el_nf) > 0, "No finite, non-fill Elevation values available for range checks"

        assert np.all((az_nf >= 0) & (az_nf <= 360)), "Non-fill Azimuth values are outside [0, 360]"
        assert np.all((el_nf >= -180) & (el_nf <= 180)), "Non-fill Elevation values are outside [-180, 180]"

    def test_non_fill_radiance_values_non_negative(self, l1b_product_dataset):
        """Radiance is a physical observable; non-fill values must be >= 0."""
        fill_value = np.float32(-999.0)
        for var_name in (
            "Filtered_Radiance_SW",
            "Filtered_Radiance_LW",
            "Filtered_Radiance_Tot",
            "Filtered_Radiance_SSW",
        ):
            vals = l1b_product_dataset[var_name].values
            non_fill = vals[vals != fill_value]
            assert np.all(non_fill >= 0), f"{var_name}: non-fill radiance values must be non-negative"

    def test_earth_sun_distance_plausible(self, l1b_product_dataset):
        """Earth-Sun distance must be within the nominal orbital range of 0.95 to 1.05 AU."""
        distance_au = float(l1b_product_dataset.attrs["Earth_Sun_Distance_AU"])
        assert 0.95 <= distance_au <= 1.05, (
            f"Earth_Sun_Distance_AU={distance_au:.4f} is outside the plausible range [0.95, 1.05]"
        )


@pytest.fixture(scope="module")
def jpss_only_l1b_product_file_path(tmp_path_factory, test_integration_data_path):
    """Run L1B algorithm in jpss_only mode with JPSS kernels only."""
    libera_files = [
        test_integration_data_path
        / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc",
        test_integration_data_path / "LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc",
        test_integration_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp",
        test_integration_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc",
    ]
    tmp_path = tmp_path_factory.mktemp("l1b_jpss_only")
    input_manifest = Manifest(
        manifest_type=ManifestType.INPUT,
        files=libera_files,
        configuration={"jpss_only": True},
    )
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )
    manifest_path = str(input_manifest.write(out_path=tmp_path))

    previous_processing_path = os.environ.get("PROCESSING_PATH")
    os.environ["PROCESSING_PATH"] = str(tmp_path)
    try:
        output_manifest_path = l1b.algorithm(manifest_path)
    finally:
        if previous_processing_path is None:
            os.environ.pop("PROCESSING_PATH", None)
        else:
            os.environ["PROCESSING_PATH"] = previous_processing_path

    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1
    return output_manifest.files[0].filename


@pytest.fixture(scope="module")
def jpss_only_l1b_product_dataset(jpss_only_l1b_product_file_path):
    """L1B product from jpss_only integration run."""
    with xr.open_dataset(jpss_only_l1b_product_file_path) as ds:
        return ds.load()


class TestL1bJpssOnlyIntegration:
    """End-to-end L1B output checks for jpss_only processing mode."""

    def test_jpss_only_geolocation_and_motor_angles(self, jpss_only_l1b_product_dataset):
        lat_fill, lon_fill, alt_fill = -999.0, -999.0, -9999.0
        lat = jpss_only_l1b_product_dataset["Latitude"].values
        lon = jpss_only_l1b_product_dataset["Longitude"].values
        alt = jpss_only_l1b_product_dataset["Altitude"].values

        non_fill_lat = lat[(lat != lat_fill) & np.isfinite(lat)]
        non_fill_lon = lon[(lon != lon_fill) & np.isfinite(lon)]
        non_fill_alt = alt[(alt != alt_fill) & np.isfinite(alt)]

        assert len(non_fill_lat) > 0
        assert len(non_fill_lon) > 0
        assert np.all((non_fill_lat >= -90) & (non_fill_lat <= 90))
        assert np.all((non_fill_lon >= -180) & (non_fill_lon <= 180))
        assert np.all(np.abs(non_fill_alt) < 5000), "Altitude should be near surface, not orbit height"

        assert np.allclose(
            jpss_only_l1b_product_dataset["Subsatellite_Latitude"].values,
            lat,
        )
        assert np.allclose(
            jpss_only_l1b_product_dataset["Subsatellite_Longitude"].values,
            lon,
        )
        assert np.allclose(
            jpss_only_l1b_product_dataset["Colatitude"].values,
            90.0 - lat,
        )
        assert np.allclose(
            jpss_only_l1b_product_dataset["Subsatellite_Colatitude"].values,
            90.0 - jpss_only_l1b_product_dataset["Subsatellite_Latitude"].values,
        )
        colat = jpss_only_l1b_product_dataset["Subsatellite_Colatitude"].values
        non_fill_colat = colat[(colat != -999.0) & np.isfinite(colat)]
        assert len(non_fill_colat) > 0
        assert np.all((non_fill_colat >= 0) & (non_fill_colat <= 180))

        assert np.all(jpss_only_l1b_product_dataset["Azimuth"].values == 0)
        assert np.all(jpss_only_l1b_product_dataset["Elevation"].values == 0)
