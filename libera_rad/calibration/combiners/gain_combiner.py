import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from cloudpathlib import AnyPath, S3Path
from libera_utils import Manifest
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.logutil import configure_task_logging

from libera_rad.calibration.combiners import l1a_combine
from libera_rad.config import cal_gain_product_definitions
from libera_rad.l1b import read_all_input_data
from libera_rad.version import version as libera_rad_version

logger = logging.getLogger(__name__)


# TODO[LIBSDC-564]: Re-evaluate shared vs event-specific steps and consolidate helpers during Tier 1.
def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """
    Main processing algorithm implementing the 7-step Libera processing workflow for gain calibration events.

    This function orchestrates the entire data processing pipeline from reading input manifests to writing output
    manifests with processed data products.

    Parameters
    ----------
    manifest_path : Path or S3Path or argparse.Namespace
        Path to the input manifest file. May be a local ``Path``, a
        cloud ``S3Path``, or an ``argparse.Namespace`` whose ``.manifest``
        attribute holds the path string.

    Returns
    -------
    Path or S3Path
        Path to the output manifest file written to the processing dropbox.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` environment variable is not set.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"l1b_{now}")

    # Step 1: Read and use the Input Manifest
    logger.info("Step 1: Reading the input manifest file")
    if isinstance(manifest_path, argparse.Namespace):
        manifest = AnyPath(manifest_path.manifest)
    else:
        manifest = AnyPath(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    all_data, _ = read_all_input_data(input_manifest)

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 3: Combine gain cal event data and calculate any required variables
    logger.info("Step 3: Creating gain calibration event dataset")
    gain_event = l1a_combine.merge_l1a_decoded_datasets(list(all_data.values()))
    gain_event.attrs["algorithm_version"] = libera_rad_version()

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    product_config_path = cal_gain_product_definitions[DataProductIdentifier.cal_gain_combined]
    output_file_path = write_libera_data_product(
        data_product_definition=product_config_path,
        data=gain_event,
        output_path=dropbox_path,
        time_variable="RAD_FULL_PACKET_ICIE_TIME",
        strict=True,
    )

    # Step 5: Create output manifest
    logger.info("Step 5: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 6: Add data files to output manifest
    logger.info(f"Step 6: Adding data files to output manifest: {output_file_path}")
    output_manifest.add_files(output_file_path.path)

    # Step 7: Write the output manifest
    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath
