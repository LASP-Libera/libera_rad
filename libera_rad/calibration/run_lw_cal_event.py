"""Runner script for making lw cal files. Will be replaced with integration test in future."""

import os
from datetime import UTC, date, datetime
from pathlib import Path

from libera_utils import Manifest, ManifestType

from libera_rad.calibration.lw_cal_combiner import algorithm


def generate_input_manifest(data_dir: Path):
    """Generating test manifest from the data in test_data"""
    # Radiometer L1A data
    rad_sample = data_dir / "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-6-1_20251213T172514_20251213T172902_R26119221206.nc"

    # Housekeeping L1A Data
    nom_hk = data_dir / "LIBERA_L1A_NOM-HK-DECODED_V5-6-1_20251213T172515_20251213T172902_R26099184809.nc"

    # Axis sample Data
    axis_sample = data_dir / "LIBERA_L1A_AXIS-SAMPLE-DECODED_V5-6-1_20251213T172514_20251213T172902_R26101035626.nc"

    # pev sw stat data
    pev = data_dir / "LIBERA_L1A_PEV-SW-STAT-DECODED_V5-6-1_20251213T172526_20251213T172830_R26101025905.nc"

    pec = data_dir / "LIBERA_L1A_PEC-SW-STAT-DECODED_V5-6-1_20251213T172515_20251213T172816_R26101031929.nc"

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_files(rad_sample, nom_hk, axis_sample, pev, pec)
    input_manifest.add_desired_time_range(
        start_datetime=datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC),
        end_datetime=datetime.combine(date.today(), datetime.max.time(), tzinfo=UTC),
    )

    input_manifest_file_path = input_manifest.write(out_path=data_dir)
    print(f"Input manifest file path: {input_manifest_file_path}")
    return input_manifest_file_path


if __name__ == "__main__":
    test_path = Path("~/Desktop/lw_cal_files")
    os.environ["PROCESSING_PATH"] = str(test_path)
    manifest = generate_input_manifest(test_path)
    algorithm(manifest)
