import pytest
from libera_utils import Manifest, ManifestType
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners.sw_combiner import algorithm
from libera_rad.config import cal_sw_product_definitions
from tests.integration.test_calibration.cal_test_helpers import (
    assert_cal_product_conformance,
    cal_desired_time_range,
    copy_cal_input_file,
    load_cal_netcdf,
)


@pytest.mark.integration
@pytest.mark.parametrize("path_type", ["Local", "S3"], indirect=True)
def test_sw_cal_algorithm_end_to_end_conforms_to_product_definition(test_l1a_cal_data_path, cal_io_paths, monkeypatch):
    input_dir, output_dir = cal_io_paths

    cal_sample = input_dir / "LIBERA_L1A_CAL-SAMPLE-DECODED_V5-6-1_20251213T172514_20251213T172902_R26119221206.nc"
    rad_sample = input_dir / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-6-1_20251213T172514_20251213T172902_R26119221206.nc"
    nom_hk = input_dir / "LIBERA_L1A_NOM-HK-DECODED_V5-6-1_20251213T172515_20251213T172902_R26099184809.nc"
    pec = input_dir / "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-6-1_20251213T172515_20251213T172816_R26101031929.nc"
    pev = input_dir / "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-6-1_20251213T172526_20251213T172830_R26101025905.nc"
    copy_cal_input_file(test_l1a_cal_data_path / "short_cal_sample.nc", cal_sample)
    copy_cal_input_file(test_l1a_cal_data_path / "short_rad_sample.nc", rad_sample)
    copy_cal_input_file(test_l1a_cal_data_path / "short_nom_hk.nc", nom_hk)
    copy_cal_input_file(test_l1a_cal_data_path / "short_pec.nc", pec)
    copy_cal_input_file(test_l1a_cal_data_path / "short_pev.nc", pev)

    manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
    manifest.add_files(cal_sample, rad_sample, nom_hk, pec, pev)
    start_datetime, end_datetime = cal_desired_time_range()
    manifest.add_desired_time_range(start_datetime=start_datetime, end_datetime=end_datetime)
    manifest_path = manifest.write(out_path=input_dir)

    monkeypatch.setenv("PROCESSING_PATH", str(output_dir))
    output_manifest_path = algorithm(manifest_path)
    output_manifest = Manifest.from_file(output_manifest_path)
    assert len(output_manifest.files) == 1

    output_file = output_manifest.files[0].filename
    dataset = load_cal_netcdf(output_file)
    assert_cal_product_conformance(
        dataset,
        cal_sw_product_definitions,
        DataProductIdentifier.cal_sw_combined,
        "SW-COMBINED",
    )
