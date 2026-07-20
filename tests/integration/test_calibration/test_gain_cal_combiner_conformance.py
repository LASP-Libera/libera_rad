import numpy as np
import pytest
import xarray as xr
from libera_utils import Manifest, ManifestType

from libera_rad.calibration.combiners.cal_combine import algorithm
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID, LIBERA_CAL_OBSID_ENV
from tests.integration.test_calibration.cal_test_helpers import (
    assert_cal_product_conformance,
    cal_desired_time_range,
    load_cal_netcdf,
    write_nom_hk_fixture,
    write_time_aligned_companion,
)


@pytest.mark.integration
@pytest.mark.parametrize("path_type", ["Local", "S3"], indirect=True)
def test_gain_cal_algorithm_end_to_end_conforms_to_product_definition(
    test_l1a_cal_data_path, cal_io_paths, monkeypatch
):
    input_dir, output_dir = cal_io_paths
    event_spec = CAL_EVENT_BY_OBSID[512]

    nom_hk_ds = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc").load()
    nom_hk_ds["ICIE__SW_OBSID_RAD"] = ("PACKET", np.full(nom_hk_ds.sizes["PACKET"], 512, dtype=np.uint16))
    t0 = np.datetime64(nom_hk_ds["PACKET_ICIE_TIME"].values.min())
    t1 = np.datetime64(nom_hk_ds["PACKET_ICIE_TIME"].values.max())
    nom_hk = write_nom_hk_fixture(nom_hk_ds, input_dir)

    rad_full = input_dir / "LIBERA_L1A_RAD-FULL-DECODED_V5-6-1_20251215T213540_20251215T213625_R26119221206.nc"
    cal_full = input_dir / "LIBERA_L1A_CAL-FULL-DECODED_V5-6-1_20251215T214030_20251215T214103_R26099184037.nc"
    write_time_aligned_companion(test_l1a_cal_data_path / "short_rad_full.nc", rad_full, t0, t1)
    write_time_aligned_companion(test_l1a_cal_data_path / "short_cal_full.nc", cal_full, t0, t1)

    manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
    manifest.add_files(rad_full, nom_hk, cal_full)
    start_datetime, end_datetime = cal_desired_time_range()
    manifest.add_desired_time_range(start_datetime=start_datetime, end_datetime=end_datetime)
    manifest_path = manifest.write(out_path=input_dir)

    monkeypatch.setenv("PROCESSING_PATH", str(output_dir))
    monkeypatch.setenv(LIBERA_CAL_OBSID_ENV, "512")
    output_manifest_path = algorithm(manifest_path)
    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1

    output_file = output_manifest.files[0].filename
    dataset = load_cal_netcdf(output_file)
    assert_cal_product_conformance(dataset, output_file, event_spec)
