"""Unified ObsID-dispatched calibration event combiner.

Selected by the ``LIBERA_CAL_OBSID`` environment variable and invoked via
``libera-rad cal-combine <manifest>``.
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

from libera_rad.calibration.combiners import gain_combiner, lw_cal_combiner, solar_cal_combiner, sw_combiner
from libera_rad.calibration.combiners.l1a_cal_event_utils import (
    confirm_obsid_matches_hk,
    extract_nom_hk_dataset,
    read_calibration_manifest_data,
)
from libera_rad.calibration.constants import CAL_EVENT_BY_OBSID, LIBERA_CAL_OBSID_ENV, CalEventSpec
from libera_rad.config import get_cal_product_definition

logger = logging.getLogger(__name__)


def resolve_cal_obsid_from_env() -> int:
    """Read and validate ``LIBERA_CAL_OBSID`` from the environment.

    Returns
    -------
    int
        Calibration ObsID.

    Raises
    ------
    ValueError
        If the variable is missing, not an integer, or not a known cal ObsID.
    """
    raw = os.getenv(LIBERA_CAL_OBSID_ENV)
    if raw is None or raw.strip() == "":
        raise ValueError(f"{LIBERA_CAL_OBSID_ENV} environment variable is not set")
    try:
        obsid = int(raw)
    except ValueError as exc:
        raise ValueError(f"{LIBERA_CAL_OBSID_ENV} must be an integer ObsID, got {raw!r}") from exc
    if obsid not in CAL_EVENT_BY_OBSID:
        known = sorted(CAL_EVENT_BY_OBSID)
        raise ValueError(f"Unknown calibration ObsID {obsid}. Known ObsIDs: {known}")
    return obsid


def _build_event_dataset(all_data: dict, event_spec: CalEventSpec):
    """Dispatch to the family merge implementation."""
    if event_spec.family == "gain":
        return gain_combiner.build_event_dataset(all_data, event_spec)
    if event_spec.family == "swc":
        return sw_combiner.build_event_dataset(all_data, event_spec)
    if event_spec.family == "lwc":
        return lw_cal_combiner.build_event_dataset(all_data, event_spec)
    if event_spec.family == "solar":
        return solar_cal_combiner.build_event_dataset(all_data, event_spec)
    raise ValueError(f"Unsupported calibration family: {event_spec.family}")


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
    event_spec = CAL_EVENT_BY_OBSID[obsid]
    logger.info(
        "Resolved %s=%d → product=%s family=%s",
        LIBERA_CAL_OBSID_ENV,
        obsid,
        event_spec.cal_product.value,
        event_spec.family,
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
    all_data = read_calibration_manifest_data(input_manifest)

    logger.info("Step 3: Confirming NOM-HK ObsID matches environment")
    nom_hk = extract_nom_hk_dataset(all_data, event_spec)
    confirm_obsid_matches_hk(nom_hk, obsid)

    logger.info("Step 4: Building %s calibration event dataset", event_spec.family)
    cal_event = _build_event_dataset(all_data, event_spec)
    # NOM-HK inputs carry their own ProductID; overwrite before product write.
    cal_event.attrs["ProductID"] = event_spec.cal_product.value

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
