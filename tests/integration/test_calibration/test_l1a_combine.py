import xarray as xr

from libera_rad.calibration.combiners.l1a_combine import merge_l1a_decoded_datasets


def test_sw_cal_l1a_combine(test_l1a_cal_data_path):
    # Load the test data
    ds_cal_sample = xr.open_dataset(test_l1a_cal_data_path / "short_cal_sample.nc")
    ds_rad_sample = xr.open_dataset(test_l1a_cal_data_path / "short_rad_sample.nc")
    ds_nom = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc")
    ds_pec = xr.open_dataset(test_l1a_cal_data_path / "short_pec.nc")
    ds_pev = xr.open_dataset(test_l1a_cal_data_path / "short_pev.nc")

    datasets = [ds_cal_sample, ds_rad_sample, ds_nom, ds_pec, ds_pev]
    merged_ds = merge_l1a_decoded_datasets(datasets)

    assert merged_ds is not None
    assert "PACKET" not in merged_ds.dims
    assert merged_ds.sizes["CAL_SAMPLE_PACKET"] == 10
    assert merged_ds.sizes["RAD_SAMPLE_PACKET"] == 10
    assert merged_ds.sizes["NOM_HK_PACKET"] == 10
    assert merged_ds.sizes["PEC_SW_STAT_PACKET"] == 10
    assert merged_ds.sizes["PEV_SW_STAT_PACKET"] == 10


def test_lw_cal_l1a_combine(test_l1a_cal_data_path):
    # Load the test data
    ds_rad_sample = xr.open_dataset(test_l1a_cal_data_path / "short_rad_sample.nc")
    ds_nom = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc")
    ds_pec = xr.open_dataset(test_l1a_cal_data_path / "short_pec.nc")
    ds_pev = xr.open_dataset(test_l1a_cal_data_path / "short_pev.nc")

    datasets = [ds_rad_sample, ds_nom, ds_pec, ds_pev]
    merged_ds = merge_l1a_decoded_datasets(datasets)

    assert merged_ds is not None
    assert "PACKET" not in merged_ds.dims
    assert merged_ds.sizes["RAD_SAMPLE_PACKET"] == 10
    assert merged_ds.sizes["NOM_HK_PACKET"] == 10
    assert merged_ds.sizes["PEC_SW_STAT_PACKET"] == 10
    assert merged_ds.sizes["PEV_SW_STAT_PACKET"] == 10


def test_gain_cal_l1a_combine(test_l1a_cal_data_path):
    # Load the test data
    ds_rad_full = xr.open_dataset(test_l1a_cal_data_path / "short_rad_full.nc")
    ds_cal_full = xr.open_dataset(test_l1a_cal_data_path / "short_cal_full.nc")
    ds_nom = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc")

    datasets = [ds_rad_full, ds_cal_full, ds_nom]
    merged_ds = merge_l1a_decoded_datasets(datasets)

    assert merged_ds is not None
    assert "PACKET" not in merged_ds.dims
    assert merged_ds.sizes["RAD_FULL_PACKET"] == 10
    assert merged_ds.sizes["NOM_HK_PACKET"] == 10
    assert merged_ds.sizes["CAL_FULL_PACKET"] == 10


def test_solar_cal_combine(test_l1a_cal_data_path):
    # Load the test data
    ds_rad_sample = xr.open_dataset(test_l1a_cal_data_path / "short_rad_sample.nc")
    ds_nom = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc")
    ds_pev = xr.open_dataset(test_l1a_cal_data_path / "short_pev.nc")

    datasets = [ds_rad_sample, ds_nom, ds_pev]
    merged_ds = merge_l1a_decoded_datasets(datasets)

    assert merged_ds is not None
    assert "PACKET" not in merged_ds.dims
    assert merged_ds.sizes["RAD_SAMPLE_PACKET"] == 10
    assert merged_ds.sizes["NOM_HK_PACKET"] == 10
    assert merged_ds.sizes["PEV_SW_STAT_PACKET"] == 10
