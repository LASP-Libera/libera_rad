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

cal_solar_cal_product_definitions = {
    DataProductIdentifier.cal_solar_cal_face1_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "solar_cal_face1_l1a.yml")
    ),
    DataProductIdentifier.cal_solar_cal_face2_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "solar_cal_face2_l1a.yml")
    ),
    DataProductIdentifier.cal_solar_cal_face3_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "solar_cal_face3_l1a.yml")
    ),
}

cal_lw_cal_product_definitions = {
    DataProductIdentifier.cal_lw_cal_temp1_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CAL_LW_CAL_TEMP1_product_definition.yml")
    ),
    DataProductIdentifier.cal_lw_cal_temp2_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CAL_LW_CAL_TEMP2_product_definition.yml")
    ),
    DataProductIdentifier.cal_lw_cal_temp3_combined: Path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CAL_LW_CAL_TEMP3_product_definition.yml")
    ),
}
