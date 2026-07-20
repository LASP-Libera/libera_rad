"""Unit tests for solar family merge helpers."""

from unittest.mock import patch

import numpy as np
import xarray as xr

from libera_rad.calibration.combiners.solar_cal_combiner import build_event_dataset, solar_obsids
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID


def test_solar_obsids_cover_face_range():
    assert solar_obsids() == set(range(384, 396))


def test_solar_event_specs_are_per_obsid():
    assert CAL_EVENT_BY_OBSID[384].cal_product.value == "SOLAR-SSW-PRI"
    assert CAL_EVENT_BY_OBSID[389].cal_product.value == "SOLAR-TOT-SEC"


@patch("libera_rad.calibration.combiners.solar_cal_combiner.l1a_combine.merge_l1a_decoded_datasets")
@patch("libera_rad.calibration.combiners.solar_cal_combiner.l1a_cal_event_utils.select_and_slice_event_inputs")
@patch("libera_rad.calibration.combiners.solar_cal_combiner.extract_nom_hk_dataset")
def test_build_event_dataset_filters_obsid_and_sets_attrs(mock_extract, mock_select, mock_merge):
    event_spec = CAL_EVENT_BY_OBSID[385]
    times = np.array(
        ["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00"],
        dtype="datetime64[ns]",
    )
    nom_hk = xr.Dataset(
        {"ICIE__SW_OBSID_RAD": ("PACKET", np.array([2, 385, 385], dtype=np.int32))},
        coords={"PACKET_ICIE_TIME": ("PACKET", times)},
    )
    mock_extract.return_value = nom_hk
    mock_select.return_value = [nom_hk.isel(PACKET=[1, 2])]
    mock_merge.return_value = xr.Dataset()

    result = build_event_dataset({"a.nc": xr.Dataset()}, event_spec)

    assert result.attrs["solar_cal_face"] == 1
    assert result.attrs["event_pass_index"] == 1
    assert result.attrs["source_obsids"] == [385]
    assert "algorithm_version" in result.attrs
    selected_nom_hk = mock_select.call_args.kwargs["nom_hk"]
    assert selected_nom_hk.sizes["PACKET"] == 2
    assert set(selected_nom_hk["ICIE__SW_OBSID_RAD"].values.tolist()) == {385}
