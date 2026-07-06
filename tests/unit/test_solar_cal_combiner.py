"""Unit tests for solar_cal_combiner helper behavior."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners.solar_cal_combiner import (
    get_product_definition_for_solar_cal_face,
    get_solar_cal_face,
)
from libera_rad.config import cal_solar_product_definitions


def _all_data_for_obsids(obsids: list[int]) -> dict[str, xr.Dataset]:
    nom_hk = xr.Dataset({"ICIE__SW_OBSID_RAD": ("PACKET", np.array(obsids, dtype=np.int32))})
    return {"LIBERA_L1A_NOM-HK-DECODED_V5-4-2_20251120T175950_20251120T190549_R26016183821.nc": nom_hk}


@pytest.mark.parametrize(
    ("obsid", "expected_face"),
    [
        (384, DataProductIdentifier.cal_solar_face1_combined),
        (385, DataProductIdentifier.cal_solar_face1_combined),
        (388, DataProductIdentifier.cal_solar_face2_combined),
        (389, DataProductIdentifier.cal_solar_face2_combined),
        (392, DataProductIdentifier.cal_solar_face3_combined),
        (393, DataProductIdentifier.cal_solar_face3_combined),
    ],
)
def test_get_solar_cal_face_returns_expected_identifier(obsid, expected_face):
    assert get_solar_cal_face(_all_data_for_obsids([obsid])) == expected_face


def test_get_solar_cal_face_raises_for_multiple_faces():
    with pytest.raises(ValueError, match="more than one solar-cal face"):
        get_solar_cal_face(_all_data_for_obsids([384, 388]))


@pytest.mark.parametrize("face_identifier", list(cal_solar_product_definitions))
def test_get_product_definition_for_solar_cal_face(face_identifier):
    result = get_product_definition_for_solar_cal_face(face_identifier)
    assert isinstance(result, Path)
    assert result == cal_solar_product_definitions[face_identifier]
