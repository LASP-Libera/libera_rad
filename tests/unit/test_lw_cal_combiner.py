"""Unit tests for LWC family merge."""

from unittest.mock import patch

import xarray as xr

from libera_rad.calibration.combiners.lw_cal_combiner import build_event_dataset
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID


@patch("libera_rad.calibration.combiners.lw_cal_combiner.l1a_combine.merge_l1a_decoded_datasets")
@patch("libera_rad.calibration.combiners.lw_cal_combiner.l1a_cal_event_utils.select_and_slice_event_inputs")
def test_build_event_dataset_sets_algorithm_version(mock_select, mock_merge):
    event_spec = CAL_EVENT_BY_OBSID[320]
    mock_select.return_value = [xr.Dataset()]
    mock_merge.return_value = xr.Dataset()
    all_data = {"a.nc": xr.Dataset()}
    result = build_event_dataset(all_data, event_spec)
    assert "algorithm_version" in result.attrs
    mock_select.assert_called_once_with(all_data, event_spec)
    mock_merge.assert_called_once()
