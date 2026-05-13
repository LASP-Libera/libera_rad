import os
from pathlib import Path

import pytest
import xarray as xr
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.io.product_definition import LiberaVariableDefinition

from libera_rad.calibration.l1a_combine import merge_l1a_decoded_datasets
from libera_rad.config import CAL_SW_COMBINED_PRODUCT_DEFINITION_PATH
from libera_rad.version import version as libera_rad_version


def _set_dimensions_definition_override() -> None:
    dimensions_definition_path = (
        Path(__file__).resolve().parents[4] / "libera_utils" / "libera_utils" / "data" / "libera_dimensions.yml"
    )
    if dimensions_definition_path.exists():
        os.environ["LIBERA_DIMENSIONS_DEFINITION_PATH"] = str(dimensions_definition_path)
        # Product-definition dimension validation caches allowed names across tests.
        # Reset so this test consistently reloads from the override path in full-suite runs.
        LiberaVariableDefinition._standard_allowed_dimensions = {}


def _build_sw_cal_merged_dataset(file_paths: list[Path]) -> xr.Dataset:
    datasets = [xr.open_dataset(path).load() for path in file_paths]
    merged = merge_l1a_decoded_datasets(datasets)
    merged.attrs["ProductID"] = "SW-CAL-COMBINED"
    merged.attrs["algorithm_version"] = libera_rad_version()
    return merged


@pytest.mark.integration
def test_sw_cal_merge_conforms_to_product_definition(test_l1a_cal_data_path):
    _set_dimensions_definition_override()

    file_paths = [
        test_l1a_cal_data_path / "short_axis_sample.nc",
        test_l1a_cal_data_path / "short_cal_sample.nc",
        test_l1a_cal_data_path / "short_rad_sample.nc",
        test_l1a_cal_data_path / "short_nom_hk.nc",
        test_l1a_cal_data_path / "short_pec.nc",
        test_l1a_cal_data_path / "short_pev.nc",
    ]
    merged = _build_sw_cal_merged_dataset(file_paths)

    definition = LiberaDataProductDefinition.from_yaml(CAL_SW_COMBINED_PRODUCT_DEFINITION_PATH)
    merged = definition.enforce_dataset_conformance(merged)
    errors = definition.check_dataset_conformance(merged, strict=False)

    assert errors == [], "\n".join(errors[:30])


@pytest.mark.integration
def test_sw_cal_merge_full_inputs_when_available(test_data_path):
    _set_dimensions_definition_override()

    full_input_dir = test_data_path / "sw_cal_full_inputs"
    if not full_input_dir.exists():
        pytest.skip("No full SW-CAL decoded input dataset directory available at tests/test_data/sw_cal_full_inputs.")

    required_names = [
        "axis_sample.nc",
        "cal_sample.nc",
        "rad_sample.nc",
        "nom_hk.nc",
        "pec_sw_stat.nc",
        "pev_sw_stat.nc",
    ]
    file_paths = [full_input_dir / file_name for file_name in required_names]
    missing = [str(path) for path in file_paths if not path.exists()]
    if missing:
        pytest.skip(f"Full SW-CAL decoded inputs missing: {missing}")

    merged = _build_sw_cal_merged_dataset(file_paths)

    definition = LiberaDataProductDefinition.from_yaml(CAL_SW_COMBINED_PRODUCT_DEFINITION_PATH)
    merged = definition.enforce_dataset_conformance(merged)
    errors = definition.check_dataset_conformance(merged, strict=False)

    assert errors == [], "\n".join(errors[:30])
