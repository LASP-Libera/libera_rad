"""Module containing paths to configuration and reference data for L1B processing."""

import os
from pathlib import Path

import yaml
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.product_definition import LiberaDataProductDefinition

from libera_rad.calibration.constants import CalEventSpec

transfer_function_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transfer_function.nc")
data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

l1b_ground_calibration_path = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l1b_ground_calibration.json")
)

product_config_path = data_path / "L1B_RAD-4CH_product_definition.yml"

#: Calibration product definition YAML template per family, keyed by the family TRIMMED
#: ProductID from ``libera_utils.obsids``. One template serves every ObsID in the family;
#: ``ProductID`` is overridden with the ObsID's own CAL product at write time.
CAL_FAMILY_PRODUCT_DEFINITIONS: dict[DataProductIdentifier, Path] = {
    DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed: data_path / "GAIN_product_definition.yml",
    DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed: data_path / "SWC_product_definition.yml",
    DataProductIdentifier.l1a_icie_nom_hk_lwc_family_trimmed: data_path / "LWC_product_definition.yml",
    DataProductIdentifier.l1a_icie_nom_hk_solar_family_trimmed: data_path / "SOLAR_product_definition.yml",
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
    family_path = CAL_FAMILY_PRODUCT_DEFINITIONS[event_spec.trimmed_product]
    with family_path.open("r") as handle:
        yaml_data = yaml.safe_load(handle)
    yaml_data.setdefault("attributes", {})["ProductID"] = event_spec.cal_product.value
    return LiberaDataProductDefinition(**yaml_data)
