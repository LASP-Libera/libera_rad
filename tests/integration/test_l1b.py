"""Integration Test for the l1b algorithm"""
import numpy as np
import xarray as xr
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest

from libera_rad import l1b


class TestL1b:
    def test_l1b_manifest(self, generate_input_manifest, monkeypatch, tmp_path, test_integration_data_path):
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest_path = l1b.algorithm(generate_input_manifest)

        # Compare manifest contents
        actual = Manifest.from_file(output_manifest_path)
        # Checksums, UIDs, and full paths will be different, but format should be the same and file names the same
        assert len(actual.files) == 1
        assert actual.manifest_type == "OUTPUT"
        assert LiberaDataProductFilename.from_file_path(output_manifest_path)

    def test_l1b_product(self, generate_input_manifest, monkeypatch, tmp_path, test_integration_data_path):
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest_path = l1b.algorithm(generate_input_manifest)
        expected_dataset_path = (test_integration_data_path /
                                 "LIBERA_L1B_RAD-4CH_V0-4-4_20251120T175950_20251120T190549_R26019201635.nc")
        expected_libera_file_name = LiberaDataProductFilename.from_file_path(expected_dataset_path)

        # Compare output file contents
        expected = xr.open_dataset(expected_dataset_path)
        output_manifest = Manifest.from_file(output_manifest_path)

        for file in output_manifest.files:
            if (LiberaDataProductFilename.from_file_path(file.filename).data_product_id
                    == expected_libera_file_name.data_product_id):
                actual = xr.open_dataset(file.filename)
                assert set(actual.variables) == set(expected.variables)

                for var_name in set(expected.variables):
                    actual_vals = actual[var_name].values
                    expected_vals = expected[var_name].values
                    if np.issubdtype(actual_vals.dtype, np.floating):
                        assert np.allclose(actual_vals, expected_vals, rtol=1e-3, atol=1e-3, equal_nan=True)
                    else:
                        assert np.array_equal(actual_vals, expected_vals)
