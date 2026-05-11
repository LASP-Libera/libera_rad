import pytest
from libera_utils import Manifest, ManifestType
from libera_utils.constants import DataProductIdentifier

from libera_rad.calibration.combiners.gain_combiner import algorithm
from libera_rad.config import cal_gain_product_definitions
from tests.integration.test_calibration.cal_test_helpers import (
    assert_cal_product_conformance,
    cal_desired_time_range,
    copy_cal_input_file,
    load_cal_netcdf,
)


@pytest.mark.integration
@pytest.mark.parametrize("path_type", ["Local", "S3"], indirect=True)
def test_gain_cal_algorithm_end_to_end_conforms_to_product_definition(
    test_l1a_cal_data_path, cal_io_paths, monkeypatch
):
    input_dir, output_dir = cal_io_paths

    rad_full = input_dir / "LIBERA_L1A_RAD-FULL-DECODED_V5-6-1_20251215T213540_20251215T213625_R26119221206.nc"
    cal_full = input_dir / "LIBERA_L1A_CAL-FULL-DECODED_V5-6-1_20251215T214030_20251215T214103_R26099184037.nc"
    nom_hk = input_dir / "LIBERA_L1A_NOM-HK-DECODED_V5-6-1_20251215T213540_20251215T213822_R26099184809.nc"
    copy_cal_input_file(test_l1a_cal_data_path / "short_rad_full.nc", rad_full)
    copy_cal_input_file(test_l1a_cal_data_path / "short_cal_full.nc", cal_full)
    copy_cal_input_file(test_l1a_cal_data_path / "short_nom_hk.nc", nom_hk)

    manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
    manifest.add_files(rad_full, nom_hk, cal_full)
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
        cal_gain_product_definitions,
        DataProductIdentifier.cal_gain_combined,
        "GAIN-COMBINED",
    )
