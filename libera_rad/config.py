"""Module containing paths to configuration and reference data for L1B processing."""

import os
from pathlib import Path

from libera_utils.constants import DataProductIdentifier

transfer_function_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transfer_function.nc")

l1b_ground_calibration_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l1b_ground_calibration.json")
)

product_config_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "L1B_RAD-4CH_product_definition.yml")
)

cal_lw_cal_product_definitions = {
    DataProductIdentifier.cal_lw_cal_temp1_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LW_CAL_TEMP1_COMBINED_product_definition.yml")
    ),
    DataProductIdentifier.cal_lw_cal_temp2_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LW_CAL_TEMP2_COMBINED_product_definition.yml")
    ),
    DataProductIdentifier.cal_lw_cal_temp3_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LW_CAL_TEMP3_COMBINED_product_definition.yml")
    ),
}

cal_gain_product_config_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GAIN_CAL_COMBINED_product_definition.yml")
)

CAL_SW_COMBINED_PRODUCT_DEFINITION_PATH = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SW-CAL-COMBINED_product_definition.yml")
)
