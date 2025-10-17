"""L1b processing code libera RAD camera"""

# Standard
import argparse
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

# installed
import xarray as xr
from cloudpathlib import AnyPath, S3Path
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import (
    LiberaDataProductFilename,
    ManifestFilename,
    format_semantic_version,
)
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_copy_file, smart_open

from libera_rad.version import version

logger = logging.getLogger(__name__)


# TODO [LIBSDC-660]: Update this method with proper error handling when reading the
#  expected L1A files as input
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

    logger.info(f"Reading the input manifest file: {parsed_cli_args.manifest}")
    input_manifest = Manifest.from_file(parsed_cli_args.manifest)

    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    for file in input_manifest.files:
        try:
            incoming_file = file.filename
            with smart_open(incoming_file):
                logger.info(f"Successfully opened file: {incoming_file}")
        except Exception as excep:
            logger.info(f"Unsuccessfully opened the file {file.filename}")
            raise excep

        data_product_file = write_data_product(file.filename, input_manifest.filename)

        output_manifest.add_files(data_product_file)
        time.sleep(1)

    dropbox_path = os.getenv("PROCESSING_PATH")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Wrote the output manifest to {dropbox_path}")

    return output_manifest_filepath


# TODO [LIBSDC-662]: Update this using libera_utils DataProductDefinition
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

    logger.info(f"Opening the file {incoming_file}")
    with smart_open(incoming_file) as file_handle:
        byte_data = file_handle.read()
        incoming_data = xr.open_dataset(byte_data, engine="netcdf4")

        incoming_data.attrs["Incoming_Process_Date(UTC)"] = str(datetime.now(UTC))
        if not isinstance(input_man, ManifestFilename):
            input_man = ManifestFilename.from_file_path(input_man)
        incoming_data.attrs["Incoming_manifest_name"] = str(input_man.path.name)

        timestamp = datetime.now(UTC)

        data_id = DataProductIdentifier.l1b_rad
        dropbox_path = AnyPath(os.getenv("PROCESSING_PATH"))
        data_product_filename = LiberaDataProductFilename.from_filename_parts(
            data_level=data_id.data_level,
            product_name=data_id.product_name,
            version=format_semantic_version(version()),
            utc_start=datetime(2027, 1, 1, 00, 00, 00, tzinfo=UTC),
            utc_end=datetime(2027, 1, 1, 23, 59, 59, tzinfo=UTC),
            revision=timestamp,
        )

        incoming_data.to_netcdf(
            data_product_filename.path.name,
            mode="w",
            engine="netcdf4",
        )
        output_location = dropbox_path / data_product_filename.path.name
        smart_copy_file(data_product_filename.path.name, output_location, delete=True)

        logger.info(f"Wrote output file to dropbox at {output_location}")

        return output_location
