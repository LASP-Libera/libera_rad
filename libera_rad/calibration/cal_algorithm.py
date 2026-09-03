"""ObsID-dispatched calibration product algorithm.

Selected by the ``LIBERA_CAL_OBSID`` environment variable and invoked via
``libera-rad cal-combine <manifest>``.

SWC/LWC/SOLAR events always attach SPICE-derived Azimuth/Elevation. The AZROT-CK
and ELSCAN-CK kernels are generated here, per event, from the AXIS-SAMPLE-DECODED
L1A input trimmed to the NOM-HK event window — cal-combine takes no SPICE inputs
and ignores any kernels on the manifest. Unlike L1B, it does not honor
``configuration.use_geo``.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from cloudpathlib import AnyPath, S3Path
from libera_utils import Manifest
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.logutil import configure_task_logging

from libera_rad.calibration.combiners.l1a_cal_event_utils import (
    add_input_files,
    attach_azimuth_elevation_from_axis_sample,
    build_event_dataset,
    confirm_obsid_matches_hk,
    extract_kernel_source_dataset,
    extract_nom_hk_dataset,
    family_needs_azimuth_elevation_positions,
    nom_hk_event_window,
    read_all_cal_input_data,
)
from libera_rad.calibration.constants import LIBERA_CAL_OBSID_ENV, get_cal_event_spec
from libera_rad.config import get_cal_product_definition

logger = logging.getLogger(__name__)


def resolve_cal_obsid_from_env() -> int:
    """Read the ``LIBERA_CAL_OBSID`` ObsID from the environment.

    Returns the number only; :func:`~libera_rad.calibration.constants.get_cal_event_spec`
    decides whether cal-combine can dispatch it, and
    :func:`~libera_rad.calibration.combiners.l1a_cal_event_utils.confirm_obsid_matches_hk`
    checks it against the ObsID the NOM-HK data actually carries.

    Returns
    -------
    int
        Calibration ObsID.

    Raises
    ------
    ValueError
        If the variable is missing or is not an integer.
    """
    raw = os.getenv(LIBERA_CAL_OBSID_ENV)
    if raw is None or raw.strip() == "":
        raise ValueError(f"{LIBERA_CAL_OBSID_ENV} environment variable is not set")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{LIBERA_CAL_OBSID_ENV} must be an integer ObsID, got {raw!r}") from exc


def algorithm(manifest_path: Path | S3Path | argparse.Namespace) -> Path | S3Path:
    """Run ObsID-dispatched calibration combine from an input manifest.

    Parameters
    ----------
    manifest_path : Path or S3Path or argparse.Namespace
        Path to the input manifest, or a Namespace with a ``.manifest`` attribute.

    Returns
    -------
    Path or S3Path
        Path to the written output manifest.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"cal_combine_{now}")

    obsid = resolve_cal_obsid_from_env()
    event_spec = get_cal_event_spec(obsid)
    logger.info(
        "Resolved %s=%d → product=%s family=%s",
        LIBERA_CAL_OBSID_ENV,
        obsid,
        event_spec.cal_product.value,
        event_spec.trimmed_product.value,
    )

    logger.info("Step 1: Reading the input manifest file")
    if isinstance(manifest_path, argparse.Namespace):
        manifest = AnyPath(manifest_path.manifest)
    else:
        manifest = AnyPath(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info("Loaded manifest with %d files", len(input_manifest.files))

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    logger.info("Step 2: Reading all input data from manifest files")
    all_data = read_all_cal_input_data(input_manifest)

    logger.info("Step 3: Confirming NOM-HK ObsID matches environment")
    nom_hk = extract_nom_hk_dataset(all_data, event_spec)
    confirm_obsid_matches_hk(nom_hk, obsid)

    logger.info("Step 4: Building %s calibration event dataset", event_spec.trimmed_product.value)
    cal_event = build_event_dataset(all_data, event_spec)
    # NOM-HK inputs carry their own ProductID; overwrite before product write.
    cal_event.attrs["ProductID"] = event_spec.cal_product.value

    if family_needs_azimuth_elevation_positions(event_spec.trimmed_product):
        logger.info("Step 4b: Generating event AZROT/ELSCAN CKs and attaching Az/El")
        axis_file_name, axis_sample = extract_kernel_source_dataset(all_data, event_spec)
        t0, t1 = nom_hk_event_window(nom_hk)
        cal_event = attach_azimuth_elevation_from_axis_sample(cal_event, axis_sample, t0, t1)
        # The AXIS-SAMPLE granule is the real input; the kernels built from it are intermediates
        # that this run creates and discards, so they are not product provenance.
        cal_event = add_input_files(cal_event, [axis_file_name])

    logger.info("Step 5: Writing data product %s", event_spec.cal_product.value)
    product_definition = get_cal_product_definition(event_spec)
    output_file_path = write_libera_data_product(
        data_product_definition=product_definition,
        data=cal_event,
        output_path=dropbox_path,
        time_variable=event_spec.time_variable,
        strict=True,
    )

    logger.info("Step 6: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
    output_manifest.add_files(output_file_path.path)

    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info("Output manifest written to: %s", output_manifest_filepath)

    return output_manifest_filepath
