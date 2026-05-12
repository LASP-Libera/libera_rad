import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path
from libera_utils import Manifest
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.logutil import configure_task_logging

from libera_rad.calibration import l1a_combine
from libera_rad.config import cal_lw_cal_product_definitions
from libera_rad.l1b import extract_input_dataset, read_all_input_data

logger = logging.getLogger(__name__)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """
    Main processing algorithm implementing the 7-step Libera processing workflow for longwave calibration events.

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
    ValueError
        If the input data contains zero or more than one LW calibration
        event type, as determined by ``get_lw_event_type``.
    ValueError
        If no product definition is found for the detected LW calibration
        event identifier, as determined by
        ``get_product_definition_for_lw_cal_event``.
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
    all_data, spice_directory = read_all_input_data(input_manifest)
    lw_event_identifier = get_lw_event_type(all_data)

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 3: Combine lw cal event data and calculate any required variables
    logger.info("Step 3: Creating longwave calibration event dataset")
    lw_cal_event = l1a_combine.merge_l1a_decoded_datasets(list(all_data.values()))

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    product_config_path = get_product_definition_for_lw_cal_event(lw_event_identifier)
    output_file_path = write_libera_data_product(
        data_product_definition=product_config_path,
        data=lw_cal_event,
        output_path=dropbox_path,
        time_variable="radiometer_time",
        strict=True,
    )

    # Step 6: Create output manifest
    logger.info("Step 5: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 7: Add data files to output manifest
    logger.info(f"Step 6: Adding data files to output manifest: {output_file_path}")
    output_manifest.add_files(output_file_path.path)

    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def get_lw_event_type(all_data: dict[str, xr.Dataset]) -> DataProductIdentifier:
    """
    Determine the longwave calibration event type from the input datasets.

    Extracts the nominal housekeeping (HK) dataset and inspects the unique observation IDs (OBSIDs) present in the
    ``ICIE__SW_OBSID_RAD`` variable. Maps each recognized OBSID to its corresponding ``DataProductIdentifier`` and
    validates that exactly one LW calibration event type is present.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Mapping of data product identifier strings to their corresponding
        decoded L1A xarray Datasets, as returned by ``read_all_input_data``.

    Returns
    -------
    DataProductIdentifier
        The ``DataProductIdentifier`` enum member that corresponds to the
        detected longwave calibration event (one of
        ``cal_lw_cal_temp1_combined``, ``cal_lw_cal_temp2_combined``, or
        ``cal_lw_cal_temp3_combined``).

    Raises
    ------
    ValueError
        If more than one recognized LW calibration OBSID is found in the
        input data.
    ValueError
        If no recognized LW calibration OBSID is found in the input data.
    """
    obsid_event_types = {
        320: DataProductIdentifier.cal_lw_cal_temp1_combined,
        321: DataProductIdentifier.cal_lw_cal_temp2_combined,
        322: DataProductIdentifier.cal_lw_cal_temp3_combined,
    }
    nom_hk_data = extract_input_dataset(all_data, DataProductIdentifier.l1a_icie_nom_hk_decoded)
    obsids = np.unique(nom_hk_data["ICIE__SW_OBSID_RAD"].values)
    matches = []
    for obsid in obsids:
        if obsid in obsid_event_types:
            matches.append(obsid_event_types[obsid])
    if len(matches) == 1:
        logger.info(f"Longwave Calibration event detected: {matches[0]}")
        return matches[0]
    elif len(matches) > 1:
        raise ValueError("More than one longwave calibration event input data. Detected OBSIDS: " + str(obsids))
    else:
        raise ValueError("No longwave calibration events detected in input data. Detected OBSIDS: " + str(obsids))


def get_product_definition_for_lw_cal_event(event_data_product_identifier: DataProductIdentifier) -> Path:
    """
    Retrieve the product definition configuration for a LW calibration event.

    Parameters
    ----------
    event_data_product_identifier : DataProductIdentifier
        The ``DataProductIdentifier`` enum member identifying the longwave calibration event type
        (e.g. ``cal_lw_cal_temp1_combined``).

    Returns
    -------
    Path
        The product definition path associated with the given identifier, as stored in
        ``cal_lw_cal_product_definitions``.

    Raises
    ------
    ValueError
        If ``event_data_product_identifier`` is not present as a key in
        ``cal_lw_cal_product_definitions``.
    """
    product_definition = cal_lw_cal_product_definitions.get(event_data_product_identifier, None)
    if product_definition is None:
        raise ValueError(
            "No longwave calibration event detected in input manifest. Detected data product identifier: "
            + event_data_product_identifier.value
        )
    return product_definition
