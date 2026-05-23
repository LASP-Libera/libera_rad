"""L1b processing code libera RAD camera"""

import argparse
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import xarray as xr
from astropy.coordinates import get_sun
from astropy.time import Time
from cloudpathlib import AnyPath, S3Path
from libera_utils import Manifest, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.libera_spice.kernel_manager import KernelManager
from libera_utils.logutil import configure_task_logging
from numpy import ndarray

from libera_rad import geolocation
from libera_rad.calibration.constants import ChannelName
from libera_rad.config import product_config_path
from libera_rad.radiometer import radiance

logger = logging.getLogger(__name__)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """
    Main processing algorithm implementing the 7-step Libera processing workflow.

    This function orchestrates the entire data processing pipeline from reading
    input manifests to writing output manifests with processed data products.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file containing data file information.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If the PROCESSING_PATH environment variable is not set.
    Exception
        If any file cannot be opened or processed.
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
    no_geo_mode = "no_geo" in input_manifest.configuration
    if no_geo_mode:
        logger.info("No geolocation mode detected: placeholder geolocation will be used.")

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    all_input_data, dynamic_kernel_sources = read_all_input_data(input_manifest, no_geo_mode=no_geo_mode)

    # Step 3: Calculate radiometer data variables
    logger.info("Step 3: Calculating radiometer data variables")
    processed_data, dynamic_product_attributes = process_l1a_to_l1b(
        all_input_data,
        dynamic_kernel_sources,
        no_geo_mode=no_geo_mode,
    )

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    output_data_file_path = create_and_write_data_product(
        processed_data=processed_data, dynamic_attributes=dynamic_product_attributes, output_path=dropbox_path
    )

    # Step 6: Create output manifest
    logger.info("Step 5: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
    output_manifest.configuration.update(input_manifest.configuration)

    # Step 7: Add data files to output manifest
    logger.info(f"Step 6: Adding data files to output manifest: {output_data_file_path}")
    output_manifest.add_files(output_data_file_path.path)

    # Step 8: Write output manifest to output dropbox folder
    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def read_all_input_data(
    input_manifest: Manifest, no_geo_mode: bool = False
) -> tuple[dict[str, xr.Dataset], list[str] | None]:
    """
    Read and store all input data from manifest files.

    This function opens and validates all input NetCDF files from the manifest and stores them in a dictionary keyed by
    filename. SPICE kernel paths (.bc, .bsp) are collected in manifest order for
    :meth:`KernelManager.load_libera_dynamic_kernels`, which materializes each file via
    :class:`~libera_utils.libera_spice.spice_utils.KernelFileCache` (local or S3).

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest containing file information.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with filenames as keys and loaded xarray datasets as values.
    list[str] or None
        Manifest paths for dynamic SPICE kernels, in manifest order, or None
        when no_geo_mode is True and SPICE kernels are not required.

    Raises
    ------
    Exception
        If any file cannot be opened or is invalid.

    Warnings
    --------
    Logs a warning if no data files were loaded from the manifest.
    """
    logger.info("Step 2: Reading all input data from manifest files")

    all_data: dict[str, xr.Dataset] = {}
    dynamic_kernel_sources: list[str] | None = [] if not no_geo_mode else None

    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")

        try:
            if file_info.filename.endswith((".bc", ".bsp")):
                if no_geo_mode:
                    logger.info(f"No geolocation mode: skipping SPICE file {file_info.filename}")
                    continue
                dynamic_kernel_sources.append(file_info.filename)
                logger.info("Recorded SPICE kernel for KernelManager: %s", file_info.filename)
            else:
                with smart_open(file_info.filename) as file_handle:
                    LiberaDataProductFilename.from_file_path(file_info.filename)  # Ensure file is Libera Data Product
                    dataset = xr.open_dataset(file_handle).load()
                    all_data[file_info.filename] = dataset
                    logger.info(f"Successfully loaded dataset: {file_handle}")
        except Exception as e:
            logger.error(f"Failed to process file {file_info.filename}: {e}", exc_info=True)
            raise

    logger.info(
        "Successfully loaded %d datasets and %d SPICE kernel paths",
        len(all_data),
        0 if dynamic_kernel_sources is None else len(dynamic_kernel_sources),
    )

    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data, dynamic_kernel_sources


def process_l1a_to_l1b(
    all_input_data: dict[str, xr.Dataset],
    dynamic_kernel_sources: Sequence[str | Path | S3Path] | None,
    no_geo_mode: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Process L1A data and SPICE Kernels to L1B product.

    This function coordinates the full L1A to L1B processing pipeline including:
    - Loading calibration data
    - Extracting radiometer and housekeeping datasets
    - Initializing SPICE kernels for geolocation
    - Gain calibration of radiometer data
    - Downsampling calibrated radiometer data to 100Hz
    - Calculating geolocation information
    - Interpolating temperatures
    - Computing radiances
    - Packaging the final L1B product

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of input datasets keyed by filename. Expected to contain radiometer sample data ('rad_sample') and
        nominal housekeeping data ('nom_hk').
    dynamic_kernel_sources : sequence of str, pathlib.Path, or cloudpathlib.S3Path, or None
        Manifest-ordered paths to SPICE kernel files (.bc, .bsp). Each entry is materialized through
        :class:`~libera_utils.libera_spice.spice_utils.KernelFileCache` inside
        :meth:`KernelManager.load_libera_dynamic_kernels`. Not used when
        no_geo_mode is True.
    no_geo_mode : bool, optional
        When True, bypasses SPICE geolocation and uses placeholder geolocation
        values. Triggered by the presence of a 'no_geo' key in the input
        manifest configuration. Defaults to False.

    Returns
    -------
    dict[str, np.ndarray]
        L1B product data dictionary, with variables defined by the L1B product definition.

    Raises
    ------
    ValueError
        If required input datasets (radiometer or housekeeping data) are not found.
    FileNotFoundError
        If the calibration data file is not found.
    """

    # Extract input datasets
    rad_data, nom_hk_data = _extract_radiometer_datasets(all_input_data)

    # Process radiometer data: timestamps is in nanoseconds since 1958-01-01, from radiometer FPE time
    timestamps, calibrated_data_by_channel = radiance.calibrate_and_downsample_radiometer_data(rad_data)

    if no_geo_mode:
        lat_lon_alt = geolocation.create_placeholder_geolocation_dataframe(len(timestamps))
    else:
        if not dynamic_kernel_sources:
            raise ValueError("SPICE kernel sources are required for geolocation when no_geo_mode is False")
        # Initialize SPICE kernels
        with KernelManager() as km:
            km.load_libera_dynamic_kernels(dynamic_kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)

            # Calculate geolocation
            lat_lon_alt = geolocation.calculate_geolocation_for_timestamps(km, timestamps)

    # Interpolate temperatures
    interpolated_temperatures = radiance.interpolate_temperatures(timestamps, nom_hk_data)

    # Calculate radiances
    calculated_radiance_by_channel = radiance.calculate_radiances(calibrated_data_by_channel, interpolated_temperatures)

    # Package output product
    l1b_product, attributes = _package_l1b_product(timestamps, lat_lon_alt, calculated_radiance_by_channel)

    return l1b_product, attributes


def _extract_radiometer_datasets(all_input_data: dict[str, xr.Dataset]) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Extract radiometer and housekeeping datasets from input data.

    Searches through the input data dictionary to identify and extract the radiometer sample data and nominal
    housekeeping data based on filename patterns.

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of input datasets keyed by filename.

    Returns
    -------
    xr.Dataset
        Radiometer sample dataset containing raw radiometer measurements.
    xr.Dataset
        Nominal housekeeping dataset containing temperature and other ancillary measurements.

    Raises
    ------
    ValueError
        If radiometer sample data (filename containing 'rad_sample') is not found.
    ValueError
        If nominal housekeeping data (filename containing 'nom_hk') is not found.

    Notes
    -----
    Files are identified by searching for 'rad_sample' and 'nom_hk' substrings in the filename keys.
    """
    rad_data = xr.Dataset()
    nom_hk_data = xr.Dataset()

    for file_name, dataset in all_input_data.items():
        libera_filename = LiberaDataProductFilename.from_file_path(file_name)
        if libera_filename.data_product_id == DataProductIdentifier.l1a_icie_rad_sample_decoded.value:
            rad_data = dataset
        elif libera_filename.data_product_id == DataProductIdentifier.l1a_icie_nom_hk_decoded.value:
            nom_hk_data = dataset

    if not rad_data:
        raise ValueError("No radiometer sample data found in input files")
    if not nom_hk_data:
        raise ValueError("No nominal housekeeping data found in input files")

    return rad_data, nom_hk_data


def _package_l1b_product(
    timestamps: np.ndarray, lat_lon_alt: pd.DataFrame, calculated_radiance_by_channel: dict[str, np.ndarray]
) -> tuple[dict[str, ndarray], dict[str, str]]:
    """
    Package L1B product according to product definition.

    Parameters
    ----------
    timestamps : np.ndarray
        Radiometer measurement timestamps at 100Hz.
    lat_lon_alt : pd.DataFrame
        Geolocation data containing latitude, longitude, and altitude.
    calculated_radiance_by_channel : dict[str, np.ndarray]
        Calculated radiance values for each channel.

    Returns
    -------
    Tuple(dict[str, np.ndarray], dict[str, str])
        Complete L1B product dictionary with all required variables and dynamic attributes for the dataset.

    Notes
    -----
    Many fields are currently filled with placeholder values (0, -999, -9999, or -128) pending implementation of full
    product variables.
    """
    data_length = len(timestamps)
    placeholder_zeros = calculate_data_quality_flags(data_length)
    placeholder_neg999 = np.full(shape=data_length, fill_value=-999, dtype=np.float32)
    placeholder_neg9999 = np.full(shape=data_length, fill_value=-9999, dtype=np.float32)
    placeholder_neg9999_f64 = np.full(shape=data_length, fill_value=-9999, dtype=np.float64)
    placeholder_3d_neg999 = np.full(shape=[data_length, 3], fill_value=-999, dtype=np.float64)
    placeholder_3d_neg9999 = np.full(shape=[data_length, 3], fill_value=-9999, dtype=np.float64)
    placeholder_3d_neg999_f32 = np.full(shape=[data_length, 3], fill_value=-999, dtype=np.float32)
    placeholder_hourly_3d_neg999 = np.full(shape=[24, 3], fill_value=-999, dtype=np.float64)  # 24 hours per day
    placeholder_hourly_3d_neg9999 = np.full(shape=[24, 3], fill_value=-9999, dtype=np.float64)  # 24 hours per day
    radiometer_time = timestamps.astype("datetime64[ns]")

    l1b_dataset = {
        "radiometer_time": radiometer_time,
        # Position
        "Latitude": lat_lon_alt["lat"].to_numpy().astype(np.float32),
        "Terrain_Corrected_Latitude": placeholder_neg999,
        "Colatitude": placeholder_neg999,
        "Longitude": lat_lon_alt["lon"].to_numpy().astype(np.float32),
        "Terrain_Corrected_Longitude": placeholder_neg999,
        "Altitude": lat_lon_alt["alt"].to_numpy().astype(np.float32),
        "Terrain_Corrected_Altitude": placeholder_neg9999,
        "Subsatellite_Latitude": placeholder_neg999,
        "Subsolar_Latitude": placeholder_neg999,
        "Subsatellite_Colatitude": placeholder_neg999,
        "Subsolar_Colatitude": placeholder_neg999,
        "Subsatellite_Longitude": placeholder_neg999,
        "Subsolar_Longitude": placeholder_neg999,
        # Geometry
        "Along_Track_Angle": placeholder_neg999,
        "Cross_Track_Angle": placeholder_neg999,
        "Solar_Zenith_Surface": placeholder_neg999,
        "Relative_Azimuth_Surface": placeholder_neg999,
        "Viewing_Zenith_Surface": placeholder_neg999,
        "Viewing_Azimuth_Surface_WRT_North": placeholder_neg999,
        "Satellite_Position": placeholder_3d_neg9999,
        "Satellite_Position_Start_Of_Hour": placeholder_hourly_3d_neg9999,
        "Satellite_Velocity": placeholder_3d_neg999,
        "Satellite_Velocity_Start_Of_Hour": placeholder_hourly_3d_neg999,
        "Satellite_Attitude_Q0": placeholder_neg999,
        "Satellite_Attitude_Q1": placeholder_neg999,
        "Satellite_Attitude_Q2": placeholder_neg999,
        "Satellite_Attitude_Q3": placeholder_neg999,
        "Azimuth": placeholder_neg999,
        "Elevation": placeholder_neg999,
        "Line_Of_Sight": placeholder_3d_neg999_f32,
        "Radius_of_Satellite_from_Center_of_Earth": placeholder_neg9999_f64,
        "Cone_Angle": placeholder_neg999,
        "Cone_Angle_Rate": placeholder_neg999,
        "Clock_Angle": placeholder_neg999,
        "Clock_Angle_Rate": placeholder_neg999,
        # Instrument
        "Operational_Mode": np.full(shape=data_length, fill_value=-128, dtype=np.int8),  # radiometer OBSID?
        "Filtered_Radiance_SW": calculated_radiance_by_channel.get(
            ChannelName.SHORTWAVE.value, placeholder_neg999
        ).astype(np.float32),
        "Filtered_Radiance_SW_Uncertainty": placeholder_neg999,
        "Filtered_Radiance_LW": calculated_radiance_by_channel.get(
            ChannelName.LONGWAVE.value, placeholder_neg999
        ).astype(np.float32),
        "Filtered_Radiance_LW_Uncertainty": placeholder_neg999,
        "Filtered_Radiance_Tot": calculated_radiance_by_channel.get(ChannelName.TOTAL.value, placeholder_neg999).astype(
            np.float32
        ),
        "Filtered_Radiance_Tot_Uncertainty": placeholder_neg999,
        "Filtered_Radiance_SSW": calculated_radiance_by_channel.get(
            ChannelName.SPLIT_SHORTWAVE.value, placeholder_neg999
        ).astype(np.float32),
        "Filtered_Radiance_SSW_Uncertainty": placeholder_neg999,
        "Quality_Flag": placeholder_zeros,
    }

    # Calculate earth sun distance based on center time point
    middle_time = int(data_length / 2)
    time = Time(radiometer_time[middle_time], format="datetime64", scale="utc")

    sun = get_sun(time)  # Get Sun's position (GCRS frame)
    distance = sun.distance.to(u.AU)  # Get the distance in AU

    dynamic_attributes = {
        "Earth_Sun_Distance_AU": distance,
    }
    return l1b_dataset, dynamic_attributes


def calculate_data_quality_flags(data_length: int) -> np.ndarray:
    """
    Calculate data quality flags for L1B product.

    Placeholder function that currently returns an array of zeros indicating all data passes quality checks.

    Parameters
    ----------
    data_length : int
        Number of data points in the L1B product.

    Returns
    -------
    np.ndarray
        Array of quality flags with shape (data_length,). Currently all zeros.
    """
    # TODO[LIBSDC-712]: Create function for assigning data quality flags
    placeholder_zeros = np.zeros(data_length, dtype=np.uint32)
    return placeholder_zeros


def create_and_write_data_product(
    processed_data: dict[str, np.ndarray], dynamic_attributes: dict[str, str], output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """
    Store science data with metadata and write to output folder.

    This function creates a properly formatted NetCDF data product using libera_utils, which handles metadata
    management and file formatting according to Libera standards.

    Parameters
    ----------
    processed_data : dict[str, np.ndarray]
        Dictionary of processed science variables. Keys are variable names matching the product definition, values are
        numpy arrays containing the data.
    dynamic_attributes : dict[str, str]
        Dictionary of dynamic attributes to be added to the whole dataset, with keys matching the product definition.
    output_path : str | Path | S3Path
        The directory location where the output file will be written.

    Returns
    -------
    LiberaDataProductFilename
        Object containing the path to the written data product file with proper Libera naming conventions.

    Raises
    ------
    FileNotFoundError
        If the product definition file 'L1B_RAD-4CH_product_definition.yml' is not found in the data directory.

    Notes
    -----
    This function:
    - Loads the product definition from the data folder
    - Creates a DataProductConfig object with proper metadata
    - Adds the processed science data to each variable
    - Writes the data product with proper Libera naming conventions
    - Uses 'radiometer_time' as the time coordinate variable
    """
    logger.info("Steps 4: Creating and writing data product")

    # Get the product definition file path from the data folder
    if not product_config_path.exists():
        raise FileNotFoundError(
            f"Product definition file not found: {product_config_path}\n"
            "Please ensure example_product_definition.yml is in the data folder."
        )
    # Step 5: Write the data product file
    logger.info("Step 5: Writing data product to environment specified file")

    output_file_path = write_libera_data_product(
        data_product_definition=product_config_path,
        data=processed_data,
        output_path=output_path,
        time_variable="radiometer_time",
        strict=True,
        dynamic_product_attributes=dynamic_attributes,
    )
    logger.info(f"Saving to {output_file_path}")
    return output_file_path
