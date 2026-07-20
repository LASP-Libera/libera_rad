"""Module containing paths to configuration and reference data for L1B processing."""

import os
from pathlib import Path

import yaml
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.product_definition import LiberaDataProductDefinition

from libera_rad.calibration.constants import CalEventSpec, CalFamily

transfer_function_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transfer_function.nc")
data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

l1b_ground_calibration_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l1b_ground_calibration.json")
)

product_config_path = data_path / "L1B_RAD-4CH_product_definition.yml"

#: Family-level calibration product definition YAML templates (ProductID overridden at write time).
CAL_FAMILY_PRODUCT_DEFINITIONS: dict[CalFamily, Path] = {
    "gain": data_path / "GAIN_product_definition.yml",
    "swc": data_path / "SWC_product_definition.yml",
    "lwc": data_path / "LWC_product_definition.yml",
    "solar": data_path / "SOLAR_product_definition.yml",
}


def get_cal_product_definition(event_spec: CalEventSpec) -> LiberaDataProductDefinition:
    """Load the family product-definition template and set ProductID for ``event_spec``.

    Parameters
    ----------
    event_spec : CalEventSpec
        ObsID-specific calibration event specification.

    Returns
    -------
    LiberaDataProductDefinition
        Product definition with ``ProductID`` set to ``event_spec.cal_product``.
    """
    family_path = CAL_FAMILY_PRODUCT_DEFINITIONS[event_spec.family]
    with family_path.open("r") as handle:
        yaml_data = yaml.safe_load(handle)
    yaml_data.setdefault("attributes", {})["ProductID"] = event_spec.cal_product.value
    return LiberaDataProductDefinition(**yaml_data)


# Backward-compatible aliases used by tests that look up a definition by product ID.
# Each maps to the family template path; callers should prefer ``get_cal_product_definition``.
cal_gain_product_definitions = {DataProductIdentifier.cal_gain: CAL_FAMILY_PRODUCT_DEFINITIONS["gain"]}
cal_sw_product_definitions = {
    DataProductIdentifier.cal_swc_365nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
    DataProductIdentifier.cal_swc_405nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
    DataProductIdentifier.cal_swc_520nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
    DataProductIdentifier.cal_swc_635nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
    DataProductIdentifier.cal_swc_840nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
    DataProductIdentifier.cal_swc_1550nm: CAL_FAMILY_PRODUCT_DEFINITIONS["swc"],
}
cal_lw_product_definitions = {
    DataProductIdentifier.cal_lwc_temp1: CAL_FAMILY_PRODUCT_DEFINITIONS["lwc"],
    DataProductIdentifier.cal_lwc_temp2: CAL_FAMILY_PRODUCT_DEFINITIONS["lwc"],
    DataProductIdentifier.cal_lwc_temp3: CAL_FAMILY_PRODUCT_DEFINITIONS["lwc"],
}
cal_solar_product_definitions = {
    DataProductIdentifier.cal_solar_ssw_pri: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_tot_pri: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_lw_pri: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_sw_pri: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_ssw_sec: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_tot_sec: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_lw_sec: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_sw_sec: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_ssw_ter: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_tot_ter: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_lw_ter: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
    DataProductIdentifier.cal_solar_sw_ter: CAL_FAMILY_PRODUCT_DEFINITIONS["solar"],
}
