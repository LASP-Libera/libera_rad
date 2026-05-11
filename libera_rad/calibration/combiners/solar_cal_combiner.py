"""Solar calibration event combiner — production algorithm.

Receives a manifest containing:
  - A NOM-HK-DECODED L1A file covering (at minimum) the solar-cal event window
    for a single face; tightly pre-cropped by the upstream runner for efficiency.
  - RAD-SAMPLE-DECODED L1A file(s) covering the same window.
  - PEV-SW-STAT-DECODED L1A file(s) covering the same window.

The combiner is self-contained: it derives the exact event time window from the
NOM-HK OBSID content and re-slices all three datasets accordingly, so it will
produce correct output even if the manifest contains over-windowed (or full
24-hour) files.  The upstream pre-cropping performed by
:mod:`libera_rad.calibration.run_solar_cal_event` is a memory/performance
optimisation, not a correctness requirement.

Slices the datasets to the NOM-HK event window (plus configurable padding),
merges all three sources via
:func:`libera_rad.calibration.l1a_combine.merge_l1a_decoded_datasets`, and
writes one solar-cal L1A output file plus an output manifest.

Pipeline role
-------------
This module is the production-facing entry point for solar calibration event
processing.  The upstream step (event detection, NOM-HK cropping, and manifest
generation) is handled by :mod:`libera_rad.calibration.run_solar_cal_event`,
which simulates what a pipeline orchestration layer (e.g. Step Functions) would
do in production.

Public interface
----------------
    algorithm(manifest_path)  ->  Path | S3Path
"""

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

from libera_rad.calibration.combiners import l1a_cal_event_utils, l1a_combine
from libera_rad.config import cal_solar_product_definitions
from libera_rad.l1b import extract_input_dataset, read_all_input_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default symmetric time padding applied around the NOM-HK event window when
#: slicing the full 24-hour RAD-SAMPLE and PEV-SW-STAT files.
DEFAULT_PAD: np.timedelta64 = np.timedelta64(5 * 60, "s")  # 5 minutes

#: Mapping of solar-cal OBSID → face-level DataProductIdentifier.
#: OBSIDs per ICD:
#:   384-387: Face 1 (primary diffuser)
#:   388-391: Face 2 (secondary diffuser)
#:   392-395: Face 3 (tertiary diffuser)
OBSID_TO_FACE_IDENTIFIER: dict[int, DataProductIdentifier] = {
    # Face 1
    384: DataProductIdentifier.cal_solar_face1_combined,
    385: DataProductIdentifier.cal_solar_face1_combined,
    386: DataProductIdentifier.cal_solar_face1_combined,
    387: DataProductIdentifier.cal_solar_face1_combined,
    # Face 2
    388: DataProductIdentifier.cal_solar_face2_combined,
    389: DataProductIdentifier.cal_solar_face2_combined,
    390: DataProductIdentifier.cal_solar_face2_combined,
    391: DataProductIdentifier.cal_solar_face2_combined,
    # Face 3
    392: DataProductIdentifier.cal_solar_face3_combined,
    393: DataProductIdentifier.cal_solar_face3_combined,
    394: DataProductIdentifier.cal_solar_face3_combined,
    395: DataProductIdentifier.cal_solar_face3_combined,
}

#: Face number (1, 2, 3) keyed by DataProductIdentifier.
FACE_IDENTIFIER_TO_FACE_NUM: dict[DataProductIdentifier, int] = {
    DataProductIdentifier.cal_solar_face1_combined: 1,
    DataProductIdentifier.cal_solar_face2_combined: 2,
    DataProductIdentifier.cal_solar_face3_combined: 3,
}

#: First OBSID for each face number (used to derive ``event_pass_index``).
FACE_BASE_OBSIDS: dict[int, int] = {1: 384, 2: 388, 3: 392}


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Main processing algorithm for the solar calibration event combiner.

    Implements the standard Libera seven-step processing workflow adapted for
    solar calibration event products.  Reads the input manifest, loads the
    pre-cropped NOM-HK file and full 24-hour RAD-SAMPLE / PEV-SW-STAT files,
    slices the latter two to the NOM-HK event window, merges all three into a
    single flat dataset, and writes one solar-cal L1A NetCDF output file plus
    an output manifest.

    Parameters
    ----------
    manifest_path : Path or S3Path
        Path to the input manifest file.  May also be an
        ``argparse.Namespace`` whose ``.manifest`` attribute holds the path.

    Returns
    -------
    Path or S3Path
        Path to the output manifest file written to the processing dropbox.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` environment variable is not set.
    ValueError
        If no recognised solar-cal OBSID is found in the NOM-HK data.
    ValueError
        If OBSIDs from more than one solar-cal face are present in the NOM-HK
        data (the input NOM-HK must be pre-cropped to a single face+pass).
    ValueError
        If no product definition YAML is configured for the detected face.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"solar_cal_{now}")

    # Step 1: Read the input manifest
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

    # Step 2: Load all input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    all_data, _dynamic_kernel_sources = read_all_input_data(input_manifest)

    # Step 3: Detect solar-cal face from NOM-HK OBSIDs
    logger.info("Step 3: Detecting solar-cal face from NOM-HK OBSIDs")
    face_identifier = get_solar_cal_face(all_data)

    # Step 4: Slice full-day datasets to the NOM-HK event window
    logger.info("Step 4: Slicing full-day datasets to NOM-HK event window")
    nom_hk = extract_input_dataset(all_data, DataProductIdentifier.l1a_icie_nom_hk_decoded)
    pev_sw = extract_input_dataset(all_data, DataProductIdentifier.l1a_pev_sw_stat_decoded)
    rad_sample = extract_input_dataset(all_data, DataProductIdentifier.l1a_icie_rad_sample_decoded)

    # Filter NOM-HK to ONLY the OBSIDs belonging to the detected face,
    # not all 12 solar-cal OBSIDs. This ensures the event window derived
    # from hk_times.min()/max() is tightly scoped to this face+pass.
    face_num = FACE_IDENTIFIER_TO_FACE_NUM[face_identifier]
    face_obsid_set = {obsid for obsid, info in OBSID_TO_FACE_IDENTIFIER.items() if info == face_identifier}
    nom_hk_event_mask = np.isin(nom_hk["ICIE__SW_OBSID_RAD"].values, list(face_obsid_set))
    nom_hk_event = nom_hk.isel(PACKET=nom_hk_event_mask)
    logger.info(
        "NOM-HK: %d / %d packets retained after filtering to face %d OBSIDs %s",
        int(nom_hk_event_mask.sum()),
        len(nom_hk_event_mask),
        face_num,
        sorted(face_obsid_set),
    )

    # build_solar_cal_event_datasets derives t0/t1 from nom_hk_event times
    # and applies pad once. Do NOT pre-pad in generate_solar_cal_manifests
    # and then pad again here — pass pad=DEFAULT_PAD only here.
    nom_hk_out, pev_sw_sliced, rad_sample_sliced = build_solar_cal_event_datasets(
        nom_hk_event, pev_sw, rad_sample, pad=DEFAULT_PAD
    )
    # Step 5: Merge all three datasets via l1a_combine
    logger.info("Step 5: Merging datasets via l1a_combine")
    solar_cal_event = l1a_combine.merge_l1a_decoded_datasets([nom_hk_out, pev_sw_sliced, rad_sample_sliced])

    # Step 5a: Populate required global attributes
    # Filter to solar-cal OBSIDs only (ICIE__SW_OBSID_RAD can contain non-solar-cal
    # values like 2 during initialisation/transition windows).
    all_obsids = [int(o) for o in np.unique(solar_cal_event["ICIE__SW_OBSID_RAD"].values)]
    source_obsids = sorted(o for o in all_obsids if o in OBSID_TO_FACE_IDENTIFIER)
    event_pass_index = min(source_obsids) - FACE_BASE_OBSIDS[face_num]
    solar_cal_event.attrs["solar_cal_face"] = face_num
    solar_cal_event.attrs["source_obsids"] = source_obsids
    solar_cal_event.attrs["event_pass_index"] = event_pass_index
    logger.info(
        "Global attributes set: solar_cal_face=%d, source_obsids=%s, event_pass_index=%d",
        face_num,
        source_obsids,
        event_pass_index,
    )

    # Step 6: Look up product definition for the detected face
    logger.info("Step 6: Looking up product definition for face: %s", face_identifier)
    product_config_path = get_product_definition_for_solar_cal_face(face_identifier)

    # Step 7: Write output product
    logger.info("Step 7: Writing solar-cal output product")
    output_file_path = write_libera_data_product(
        data_product_definition=product_config_path,
        data=solar_cal_event,
        output_path=dropbox_path,
        time_variable="NOM_HK_PACKET_ICIE_TIME",
        strict=True,
    )
    logger.info("Output file: %s", output_file_path.path)

    # Step 8: Create output manifest
    logger.info("Step 8: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
    output_manifest.add_files(output_file_path.path)

    logger.info("Step 9: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info("Output manifest written to: %s", output_manifest_filepath)

    return output_manifest_filepath


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_solar_cal_face(all_data: dict[str, xr.Dataset]) -> DataProductIdentifier:
    """Determine the solar calibration face from the loaded input datasets.

    Extracts the NOM-HK dataset and inspects the unique values of
    ``ICIE__SW_OBSID_RAD``, mapping each recognised OBSID to its face-level
    :class:`~libera_utils.constants.DataProductIdentifier`.

    Parameters
    ----------
    all_data : dict of {str : xr.Dataset}
        Mapping of filenames to decoded L1A xarray Datasets, as returned by
        :func:`libera_rad.l1b.read_all_input_data`.

    Returns
    -------
    DataProductIdentifier
        One of ``cal_solar_face1_combined``, ``cal_solar_face2_combined``,
        or ``cal_solar_face3_combined``.

    Raises
    ------
    ValueError
        If no recognised solar-cal OBSID is found in the NOM-HK data.
    ValueError
        If OBSIDs from more than one solar-cal face are present (the NOM-HK
        must be pre-cropped to a single face before calling this combiner).
    """
    nom_hk = extract_input_dataset(all_data, DataProductIdentifier.l1a_icie_nom_hk_decoded)
    obsids = np.unique(nom_hk["ICIE__SW_OBSID_RAD"].values)

    face_identifiers: set[DataProductIdentifier] = set()
    for obsid in obsids:
        if obsid in OBSID_TO_FACE_IDENTIFIER:
            face_identifiers.add(OBSID_TO_FACE_IDENTIFIER[obsid])

    if len(face_identifiers) == 1:
        face_id = next(iter(face_identifiers))
        logger.info("Solar calibration face detected: %s", face_id)
        return face_id
    if len(face_identifiers) > 1:
        raise ValueError(
            f"OBSIDs from more than one solar-cal face were found in the NOM-HK data. "
            f"Detected OBSIDs: {obsids}. "
            f"The NOM-HK file must be pre-cropped to a single face+pass before calling this combiner."
        )
    raise ValueError(
        f"No recognised solar-cal OBSIDs found in the NOM-HK data. "
        f"Detected OBSIDs: {obsids}. "
        f"Expected one of: {sorted(OBSID_TO_FACE_IDENTIFIER.keys())}."
    )


def build_solar_cal_event_datasets(
    nom_hk: xr.Dataset,
    pev_sw: xr.Dataset,
    rad_sample: xr.Dataset,
    pad: np.timedelta64 = DEFAULT_PAD,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Slice PEV-SW-STAT and RAD-SAMPLE to the NOM-HK event window.

    NOM-HK must already be filtered to only the solar-cal event packets
    (``ICIE__SW_OBSID_RAD`` in the recognised solar-cal OBSID set) before
    calling this function — see :func:`algorithm` for how that filtering is
    applied.  PEV-SW-STAT and RAD-SAMPLE are full 24-hour datasets that are
    sliced to ``[t_start − pad, t_end + pad]`` using the min/max of the
    filtered NOM-HK ``PACKET_ICIE_TIME`` coordinate.

    The returned datasets retain the original ``PACKET`` dimension name and
    are ready to be passed directly to
    :func:`libera_rad.calibration.l1a_combine.merge_l1a_decoded_datasets`.


    Parameters
    ----------
    nom_hk : xr.Dataset
        Pre-cropped NOM-HK-DECODED dataset (``PACKET`` / ``PACKET_ICIE_TIME``).
    pev_sw : xr.Dataset
        Full 24-hour PEV-SW-STAT-DECODED dataset
        (``PACKET`` / ``PACKET_ICIE_TIME``).
    rad_sample : xr.Dataset
        Full 24-hour RAD-SAMPLE-DECODED dataset
        (``PACKET`` / ``PACKET_ICIE_TIME`` + ``RAD_SAMPLE_FPE_TIME``).
    pad : np.timedelta64
        Symmetric time padding applied around the NOM-HK window when slicing
        the full-day files (default 5 minutes).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset, xr.Dataset]
        ``(nom_hk, pev_sw_sliced, rad_sample_sliced)`` — all three retain
        their original dimension names and are ready for
        ``merge_l1a_decoded_datasets``.
    """
    hk_times = nom_hk["PACKET_ICIE_TIME"].values
    t0 = hk_times.min() - pad
    t1 = hk_times.max() + pad

    logger.info(
        "Slicing to event window (±%d min padding): [%s — %s]",
        int(pad / np.timedelta64(60, "s")),
        t0,
        t1,
    )

    # Slice PEV-SW-STAT (1-D: PACKET only)
    pev_sliced = l1a_cal_event_utils.slice_dataset_to_time_window(pev_sw, t0, t1)
    logger.info(
        "PEV-SW-STAT: %d / %d packets selected",
        pev_sliced.sizes["PACKET"],
        pev_sw.sizes["PACKET"],
    )

    # Slice RAD-SAMPLE (2-D: PACKET + RAD_SAMPLE_FPE_TIME)
    rad_sliced = l1a_cal_event_utils.slice_dataset_to_time_window(
        rad_sample, t0, t1, secondary_time_dim="RAD_SAMPLE_FPE_TIME"
    )
    logger.info(
        "RAD-SAMPLE: %d / %d packets, %d / %d FPE samples selected",
        rad_sliced.sizes["PACKET"],
        rad_sample.sizes["PACKET"],
        rad_sliced.sizes["RAD_SAMPLE_FPE_TIME"],
        rad_sample.sizes["RAD_SAMPLE_FPE_TIME"],
    )

    return nom_hk, pev_sliced, rad_sliced


def get_product_definition_for_solar_cal_face(face_identifier: DataProductIdentifier) -> Path:
    """Retrieve the product definition YAML path for a solar-cal face.

    Parameters
    ----------
    face_identifier : DataProductIdentifier
        One of ``cal_solar_face1_combined``, ``cal_solar_face2_combined``,
        or ``cal_solar_face3_combined``.

    Returns
    -------
    Path
        Path to the product definition YAML, as registered in
        :data:`libera_rad.config.cal_solar_product_definitions`.

    Raises
    ------
    ValueError
        If ``face_identifier`` is not a key in
        ``cal_solar_product_definitions``.
    """
    product_definition = cal_solar_product_definitions.get(face_identifier)
    if product_definition is None:
        raise ValueError(
            f"No product definition found for solar-cal face: {face_identifier}. "
            f"Available faces: {list(cal_solar_product_definitions.keys())}."
        )
    return product_definition
