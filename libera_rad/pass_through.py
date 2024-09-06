"""Pass through processing code Libera data formats"""
# Standard
import argparse
import logging
import os
import random
from datetime import datetime
from importlib.metadata import version
from tempfile import TemporaryDirectory

# installed
import xarray as xr
from libera_utils.io.filenaming import EphemerisKernelFilename, AttitudeKernelFilename, LiberaDataProductFilename
from libera_utils.io.filenaming import format_semantic_version
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_open, smart_copy_file, is_s3

default_log_args = {
    "format": "[%(levelname)s] %(name)s - %(message)s",
    "datefmt": '%Y-%m-%d,%H:%M:%S',
    "level": logging.INFO,
    "force": True
}
logging.basicConfig(**default_log_args)

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> str:
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
            incoming_file = file['filename']
            with smart_open(incoming_file):
                logger.info('Successfully opened file')
        except:
            logger.info('Unsuccessfully opened the file')

    logger.info("Writing the new netcdf4 file to the output manifest")
    write_dummy_data_files(output_manifest, input_manifest, parsed_cli_args.processing_step)

    logger.info("Writing the physical output manifest")
    dropbox_path = os.getenv("PROCESSING_PATH")
    output_manifest_filepath = output_manifest.write(dropbox_path)

    return output_manifest_filepath


def write_dummy_data_files(output_manifest: Manifest, input_manifest: Manifest, processing_step: str):
    """Write dummy SPICE kernels for testing"""
    data = xr.DataArray([1, 2, 3, 4, 5])
    data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    data.attrs['Incoming_manifest_name'] = str(input_manifest.filename.path.name)  # pylint: disable=no-member
    dropbox_path = os.getenv("PROCESSING_PATH")

    if processing_step == "spice_jpss":
        spk_file_path, ck_file_path = write_dummy_spice_jpss_files(data, dropbox_path)
        output_manifest.add_file_to_manifest(spk_file_path)
        output_manifest.add_file_to_manifest(ck_file_path)
    elif processing_step == "spice_azel":
        az_file_path, el_file_path = write_dummy_spice_az_el_files(data, dropbox_path)
        output_manifest.add_file_to_manifest(az_file_path)
        output_manifest.add_file_to_manifest(el_file_path)
    elif processing_step == "l1b_cam":
        l1b_cam_file_path = write_dummy_l1b_files(data, dropbox_path, "CAM")
        output_manifest.add_file_to_manifest(l1b_cam_file_path)
    elif processing_step == "l1b_rad":
        l1b_rad_file_path = write_dummy_l1b_files(data, dropbox_path, "RAD")
        output_manifest.add_file_to_manifest(l1b_rad_file_path)
    else:
        raise ValueError(f"Invalid processing step {processing_step}")


def write_dummy_spice_jpss_files(data: xr.DataArray, dropbox_path: str):
    """Write dummy SPICE kernels for JPSS one ck and one spk"""

    spk_filename = EphemerisKernelFilename.from_filename_parts(
        basepath=dropbox_path,
        spk_object='JPSS',
        utc_start=datetime(2027, 1, 2, 11, 22, 33),
        utc_end=datetime(2027, 1, 2, 12, 22, 33),
        version=format_semantic_version(version('libera_rad')),  # version of this code
        revision=datetime(2028, 1, 2, 11, 22, 33))
    spk_file_path = f"{dropbox_path}/{spk_filename.path.name}"  # pylint: disable=no-member

    logger.info(f"Writing dummy SPK kernel to {spk_file_path}")
    random_data = data + random.randint(0, 100)  #nosec B311
    with smart_open(spk_file_path, 'w') as file:
        file.write(str(random_data))

    ck_filename = AttitudeKernelFilename.from_filename_parts(
        basepath=dropbox_path,
        ck_object='JPSS',
        utc_start=datetime(2027, 1, 2, 11, 22, 33),
        utc_end=datetime(2027, 1, 2, 12, 22, 33),
        version=format_semantic_version(version('libera_rad')),  # Release candidate
        revision=datetime(2028, 1, 2, 11, 22, 33))
    ck_file_path = f"{dropbox_path}/{ck_filename.path.name}"  # pylint: disable=no-member

    random_data = data + random.randint(0, 100)  #nosec B311
    logger.info(f"Writing dummy CK kernel to {ck_file_path}")
    with smart_open(ck_file_path, 'w') as file:
        file.write(str(random_data))

    return spk_file_path, ck_file_path


def write_dummy_spice_az_el_files(data: xr.DataArray, dropbox_path: str):
    """Write dummy SPICE kernels for Az an El cks"""

    az_filename = AttitudeKernelFilename.from_filename_parts(
        basepath=dropbox_path,
        ck_object='AZROT',
        utc_start=datetime(2027, 1, 2, 11, 22, 33),
        utc_end=datetime(2027, 1, 2, 12, 22, 33),
        version=format_semantic_version(version('libera_rad')),  # version of this code
        revision=datetime(2028, 1, 2, 11, 22, 33))
    az_file_path = f"{dropbox_path}/{az_filename.path.name}"  # pylint: disable=no-member

    logger.info(f"Writing dummy Az CK kernel to {az_file_path}")
    random_data = data + random.randint(0, 100)  #nosec B311
    with smart_open(az_file_path, 'w') as file:
        file.write(str(random_data))

    el_filename = AttitudeKernelFilename.from_filename_parts(
        basepath=dropbox_path,
        ck_object='ELSCAN',
        utc_start=datetime(2027, 1, 2, 11, 22, 33),
        utc_end=datetime(2027, 1, 2, 12, 22, 33),
        version=format_semantic_version(version('libera_rad')),  # Release candidate
        revision=datetime(2028, 1, 2, 11, 22, 33))
    el_file_path = f"{dropbox_path}/{el_filename.path.name}"  # pylint: disable=no-member

    random_data = data + random.randint(0, 100)  #nosec B311
    logger.info(f"Writing dummy El CK kernel to {el_file_path}")
    with smart_open(el_file_path, 'w') as file:
        file.write(str(random_data))

    return az_file_path, el_file_path


def write_dummy_l1b_files(data: xr.DataArray, dropbox_path: str, product_name: str):
    """Write dummy L1b camera data file"""

    l1b_cam_filename = LiberaDataProductFilename.from_filename_parts(
        basepath=dropbox_path,
        data_level="L1B",
        product_name=product_name,
        utc_start=datetime(2027, 1, 2, 11, 22, 33),
        utc_end=datetime(2027, 1, 2, 12, 22, 33),
        version=format_semantic_version(version('libera_rad')),  # version of this code
        revision=datetime(2028, 1, 2, 11, 22, 33))
    l1b_file_path = f"{dropbox_path}/{l1b_cam_filename.path.name}"  # pylint: disable=no-member

    random_data = data + random.randint(0, 100)  #nosec B311

    if is_s3(l1b_file_path):
        # write to local directory first
        with TemporaryDirectory() as tmp_dir_name:
            tmp_file_path = l1b_file_path.replace(dropbox_path, tmp_dir_name)
            logger.info(f"Writing dummy L1b {product_name} data to {tmp_file_path}")
            random_data.to_netcdf(tmp_file_path)
            # copy to S3
            logger.info(f"Copying L1b {tmp_file_path} to {l1b_file_path}")
            smart_copy_file(tmp_file_path, l1b_file_path)
    else:
        logger.info(f"Writing dummy L1b {product_name} data to {l1b_file_path}")
        random_data.to_netcdf(l1b_file_path)

    return l1b_file_path
