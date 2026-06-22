import numpy as np
import pytest
import xarray as xr
from libera_utils import Manifest, ManifestType
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners.solar_cal_combiner import algorithm
from libera_rad.config import cal_solar_product_definitions
from tests.integration.test_calibration.cal_test_helpers import (
    assert_cal_product_conformance,
    cal_desired_time_range,
    copy_cal_input_file,
    load_cal_netcdf,
    write_nom_hk_fixture,
)


@pytest.mark.integration
@pytest.mark.parametrize("path_type", ["Local", "S3"], indirect=True)
def test_solar_cal_algorithm_end_to_end_face1_conforms_to_product_definition(
    test_l1a_cal_data_path, cal_io_paths, monkeypatch
):
    input_dir, output_dir = cal_io_paths

    rad_sample = input_dir / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-6-1_20251213T172514_20251213T172902_R26119221206.nc"
    pev = input_dir / "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-6-1_20251213T172526_20251213T172830_R26101025905.nc"

    nom_hk_ds = xr.open_dataset(test_l1a_cal_data_path / "short_nom_hk.nc").load()
    nom_hk_ds["ICIE__SW_OBSID_RAD"] = ("PACKET", np.full(nom_hk_ds.sizes["PACKET"], 384, dtype=np.uint16))
    nom_hk = write_nom_hk_fixture(nom_hk_ds, input_dir)
    copy_cal_input_file(test_l1a_cal_data_path / "short_rad_sample.nc", rad_sample)
    copy_cal_input_file(test_l1a_cal_data_path / "short_pev.nc", pev)

    manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
    manifest.add_files(nom_hk, pev, rad_sample)
    start_datetime, end_datetime = cal_desired_time_range()
    manifest.add_desired_time_range(start_datetime=start_datetime, end_datetime=end_datetime)
    manifest_path = manifest.write(out_path=input_dir)

    monkeypatch.setenv("PROCESSING_PATH", str(output_dir))
    output_manifest_path = algorithm(manifest_path)
    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1

    output_file = output_manifest.files[0].filename
    dataset = load_cal_netcdf(output_file)
    assert dataset.attrs["ProductID"] == "SOLAR-FACE1-COMBINED"
    assert dataset.attrs["solar_cal_face"] == 1
    source_obsids = dataset.attrs["source_obsids"]
    if isinstance(source_obsids, list | tuple | np.ndarray):
        assert 384 in source_obsids
    else:
        assert int(source_obsids) == 384
    assert_cal_product_conformance(
        dataset,
        cal_solar_product_definitions,
        DataProductIdentifier.cal_solar_face1_combined,
        "SOLAR-FACE1-COMBINED",
    )
