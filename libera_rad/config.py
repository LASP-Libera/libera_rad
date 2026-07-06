"""Module containing paths to configuration and reference data for L1B processing."""

import os
from pathlib import Path

from libera_utils.constants import DataProductIdentifier

transfer_function_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transfer_function.nc")
data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

l1b_ground_calibration_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l1b_ground_calibration.json")
)

product_config_path = data_path / "L1B_RAD-4CH_product_definition.yml"

cal_solar_product_definitions = {
    DataProductIdentifier.cal_solar_face1_combined: data_path / "SOLAR_FACE1_COMBINED_product_definition.yml",
    DataProductIdentifier.cal_solar_face2_combined: data_path / "SOLAR_FACE2_COMBINED_product_definition.yml",
    DataProductIdentifier.cal_solar_face3_combined: data_path / "SOLAR_FACE3_COMBINED_product_definition.yml",
}

cal_lw_product_definitions = {
    DataProductIdentifier.cal_lw_temp1_combined: data_path / "LW_TEMP1_COMBINED_product_definition.yml",
    DataProductIdentifier.cal_lw_temp2_combined: data_path / "LW_TEMP2_COMBINED_product_definition.yml",
    DataProductIdentifier.cal_lw_temp3_combined: data_path / "LW_TEMP3_COMBINED_product_definition.yml",
}

cal_gain_product_definitions = {
    DataProductIdentifier.cal_gain_combined: data_path / "GAIN_COMBINED_product_definition.yml"
}

cal_sw_product_definitions = {DataProductIdentifier.cal_sw_combined: data_path / "SW_COMBINED_product_definition.yml"}
