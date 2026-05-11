"""Unit tests for gain calibration combiner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners.gain_combiner import algorithm
from libera_rad.config import cal_gain_product_definitions


@patch("libera_rad.calibration.combiners.gain_combiner.Manifest.output_manifest_from_input_manifest")
@patch("libera_rad.calibration.combiners.gain_combiner.write_libera_data_product")
@patch("libera_rad.calibration.combiners.gain_combiner.l1a_combine.merge_l1a_decoded_datasets")
@patch("libera_rad.calibration.combiners.gain_combiner.read_all_input_data")
@patch("libera_rad.calibration.combiners.gain_combiner.Manifest.from_file")
def test_algorithm_uses_rad_full_time_variable_and_expected_product_definition(
    mock_manifest_from_file,
    mock_read_all_input_data,
    mock_merge_l1a,
    mock_write_product,
    mock_output_manifest_from_input,
    monkeypatch,
):
    monkeypatch.setenv("PROCESSING_PATH", "/tmp")

    input_manifest = MagicMock()
    input_manifest.files = ["a", "b", "c"]
    mock_manifest_from_file.return_value = input_manifest

    mock_read_all_input_data.return_value = ({"mock.nc": xr.Dataset()}, [])
    mock_merge_l1a.return_value = xr.Dataset()
    mock_write_product.return_value = SimpleNamespace(path=Path("/tmp/mock_output.nc"))

    output_manifest = MagicMock()
    output_manifest.write.return_value = Path("/tmp/mock_manifest.json")
    mock_output_manifest_from_input.return_value = output_manifest

    with patch("libera_rad.calibration.combiners.gain_combiner.configure_task_logging"):
        output_path = algorithm(Path("/tmp/input_manifest.json"))

    assert output_path == Path("/tmp/mock_manifest.json")
    mock_write_product.assert_called_once()
    kwargs = mock_write_product.call_args.kwargs
    assert kwargs["strict"] is True
    assert kwargs["time_variable"] == "RAD_FULL_PACKET_ICIE_TIME"
    assert kwargs["data_product_definition"] == cal_gain_product_definitions[DataProductIdentifier.cal_gain_combined]
