"""Tests for the l1b algorithm"""
# Standard
from argparse import Namespace
# Installed
import xarray as xr
# Local
from libera_utils.io.manifest import Manifest
from libera_rad.l1b import algorithm


def test_algorithm(generate_input_manifest, monkeypatch, tmp_path):
    """Testing the algorithm to generate output manifests"""

    monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
    algo_inputs = Namespace(manifest=str(generate_input_manifest))
    output_manifest_path = algorithm(algo_inputs)

    output_manifest_obj = Manifest.from_file(output_manifest_path)

    for file in output_manifest_obj.files:
        data_product = xr.open_dataset(file['filename'])
        print(data_product)
