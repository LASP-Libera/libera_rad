"""Runner script for making gain cal files. Will be replaced with integration test in future."""

import os
from datetime import UTC, date, datetime
from pathlib import Path

from libera_utils import Manifest, ManifestType

from libera_rad.calibration.combiners.gain_combiner import algorithm


def generate_input_manifest(data_dir: Path):
    """Generating test manifest from the data in test_data"""
    # Radiometer L1A data
    rad_full = data_dir / "LIBERA_L1A_RAD-FULL-DECODED_V5-6-1_20251215T213540_20251215T213625_R26119221206.nc"

    # Housekeeping L1A Data
    nom_hk = data_dir / "LIBERA_L1A_NOM-HK-DECODED_V5-6-1_20251215T213540_20251215T213822_R26099184809.nc"

    # Cal Full Data
    cal_full = data_dir / "LIBERA_L1A_CAL-FULL-DECODED_V5-6-1_20251215T214030_20251215T214103_R26099184037.nc"

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_files(rad_full, nom_hk, cal_full)
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )

    input_manifest_file_path = input_manifest.write(out_path=data_dir)
    print(f"Input manifest file path: {input_manifest_file_path}")
    return input_manifest_file_path


if __name__ == "__main__":
    test_path = Path("~/Desktop/gain_cal_files").expanduser()
    os.environ["PROCESSING_PATH"] = str(test_path)
    manifest = generate_input_manifest(test_path)
    algorithm(manifest)
