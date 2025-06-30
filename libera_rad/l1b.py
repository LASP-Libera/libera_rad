"""L1b processing code libera RAD camera"""

# Standard
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# installed
import xarray as xr
from cloudpathlib import S3Path
from libera_utils.aws.constants import DataLevel
from libera_utils.io.filenaming import AbstractValidFilename, ManifestFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> Path | S3Path:
    """

    Parameters
    ----------
    parsed_cli_args: argparse.Namespace
        command line argument of the incoming manifest file


    Returns
    -------
    output_manifest: str
        the path of the output manifest as a string

    """

    logger.info("Reading the input manifest file")
    input_manifest = Manifest.from_file(parsed_cli_args.manifest)

    logger.info("Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    logger.info("Reading each file in the manifest")
    for file in input_manifest.files:
        try:
            incoming_file = file.filename
            with smart_open(incoming_file):
                logger.info("Successfully opened file")
        except Exception as excep:
            logger.info("Unsuccessfully opened the file")
            raise excep

        logger.info("Writing the new netcdf4 file to the output manifest")
        data_product_file = write_data_product(file.filename, input_manifest.filename)

        output_manifest.add_files(data_product_file)
        time.sleep(1)

    logger.info("Writing the physical output manifest")
    dropbox_path = os.getenv("PROCESSING_PATH")
    output_manifest_filepath = output_manifest.write(dropbox_path)

    return output_manifest_filepath


def write_data_product(incoming_file: str, input_man: str | ManifestFilename) -> str:
    """
    Takes a file named in the input manifest and generates the output nectdf4 file, with tags and correct output name

    Parameters
    ----------
    incoming_file: Union[str, ManifestFilename]
        incoming data file retrieved from the input manifest file
    input_man: ManifestFilename
        The name of the incoming manifest file that houses the files, needed to add tags to the
        newly created netcdf4 files


    Returns
    -------
    str
        the file path of the data product filename
    """

    logger.info("Opening the file ")
    incoming_data = xr.open_dataset(incoming_file)

    logger.info("Adding tags to the netcdf4 dataset")
    incoming_data.attrs["Incoming_Process_Date(UTC)"] = str(datetime.utcnow())
    if not isinstance(input_man, ManifestFilename):
        input_man = AbstractValidFilename.from_file_path(input_man)
    incoming_data.attrs["Incoming_manifest_name"] = str(input_man.path.name)

    timestamp = datetime.utcnow().strftime("%Y%m%dt%H%M%S")

    dropbox_path = os.getenv("PROCESSING_PATH")
    data_product_filename = (
        f"{dropbox_path}/libera_cam_{DataLevel['L1B']}_ThisIsARandDesc_{timestamp}_vM1m2p3_r27002112233.h5"
    )

    logger.info("Writing the new netcdf4 file to the output manifest")
    incoming_data.to_netcdf(data_product_filename)

    return data_product_filename
