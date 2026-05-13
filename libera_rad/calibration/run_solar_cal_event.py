"""Runner script for generating solar calibration event output files.

This script handles the full end-to-end workflow from a directory of raw
24-hour L1A files through to solar-cal combined output products.  It is
intended for development and testing against IOV data and will be replaced
by an integration test in a future ticket.

Unlike the LW cal runner, which receives pre-cropped input files, the solar
cal pipeline requires explicit NOM-HK cropping because the full 24-hour file
spans many observation types.  This script:

  1. Scans NOM-HK files for solar-cal OBSID windows and groups them into
     per-face, per-orbital-pass events (:func:`detect_solar_cal_events`).
  2. For each event, crops the three L1A products (NOM-HK, RAD-SAMPLE,
     PEV-SW-STAT) to the event window and writes them with valid Libera
     filenames (:func:`write_event_inputs`).
  3. Builds an input manifest containing those three files and calls
     ``solar_cal_combiner.algorithm()`` (:func:`generate_solar_cal_manifests`).

Module layout
-------------
``SolarCalEvent``
    Immutable record describing one detected face+pass event.
``detect_solar_cal_events``
    Pure detection step — no file writes, no side-effects.
``write_event_inputs``
    Writes the three pre-windowed files for one event.
``generate_solar_cal_manifests``
    Orchestrates detection → input preparation → manifest creation.
``_write_windowed_product``
    Internal helper: filter, slice, and write one L1A product file.
"""

import logging
import os
from datetime import UTC
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from libera_utils import Manifest, ManifestType
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename

from libera_rad.calibration import l1a_event_utils
from libera_rad.calibration.solar_cal_combiner import (
    FACE_IDENTIFIER_TO_FACE_NUM,
    OBSID_TO_FACE_IDENTIFIER,
    algorithm,
)
from libera_rad.version import version

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Symmetric time padding applied around each event window when writing the
#: pre-windowed input files.
EVENT_PAD: np.timedelta64 = np.timedelta64(5 * 60, "s")  # 5 minutes

#: Maximum time gap between consecutive OBSID windows that still belongs to
#: the same orbital pass.
PASS_GAP: np.timedelta64 = np.timedelta64(2, "h")

#: Minimum number of consecutive packets required to count as a valid event.
MIN_PACKETS: int = 10


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------


class SolarCalEvent(NamedTuple):
    """Immutable record describing one detected solar calibration event.

    Attributes
    ----------
    face_num : int
        Diffuser face number (1 = primary, 2 = secondary, 3 = tertiary).
    pass_idx : int
        Zero-based index of this orbital pass within the face.
    t_event_start : np.datetime64
        Start of the earliest OBSID run in this pass (no padding applied).
    t_event_end : np.datetime64
        End of the latest OBSID run in this pass (no padding applied).
    """

    face_num: int
    pass_idx: int
    t_event_start: np.datetime64
    t_event_end: np.datetime64


# ---------------------------------------------------------------------------
# Step 1: Event detection (pure, no I/O side-effects)
# ---------------------------------------------------------------------------


def detect_solar_cal_events(
    data_dir: Path,
    min_packets: int = MIN_PACKETS,
    pass_gap: np.timedelta64 = PASS_GAP,
) -> list[SolarCalEvent]:
    """Scan NOM-HK files and return a list of detected solar-cal events.

    Reads only the OBSID and timestamp variables (fast scan) to locate
    solar-cal OBSID windows, then groups them into per-face, per-pass events.
    No files are written and no data products are produced.

    The OBSID → face mapping is driven by
    :data:`~libera_rad.calibration.solar_cal_combiner.OBSID_TO_FACE_IDENTIFIER`
    so the detection and combiner modules stay in sync without duplicating the
    OBSID table.

    Parameters
    ----------
    data_dir : Path
        Directory containing L1A ``NOM-HK-DECODED`` ``*.nc`` files.
    min_packets : int
        Minimum consecutive packets required for a valid OBSID run (default 10).
    pass_gap : np.timedelta64
        Maximum time gap between consecutive windows still belonging to the
        same orbital pass (default 2 hours).

    Returns
    -------
    list[SolarCalEvent]
        One entry per detected face+pass, sorted by face number then pass index.
    """
    all_solar_cal_obsids = list(OBSID_TO_FACE_IDENTIFIER.keys())

    # Fast scan: read only the OBSID + time columns across all NOM-HK files
    all_windows = l1a_event_utils.scan_files_for_field_windows(
        data_dir=data_dir,
        product_token="NOM-HK-DECODED",
        target_values=all_solar_cal_obsids,
        field_var="ICIE__SW_OBSID_RAD",
        min_count=min_packets,
    )
    if not all_windows:
        logger.warning("No solar-cal OBSID windows detected in %s.", data_dir)
        return []
    logger.info("Detected %d raw OBSID window(s) across all faces.", len(all_windows))

    # Load the NOM-HK subset needed to refine per-face run detection
    nom_hk_scan = l1a_event_utils.load_l1a_product(
        data_dir, "NOM-HK-DECODED", filter_windows=all_windows
    )
    hk_obsids = nom_hk_scan["ICIE__SW_OBSID_RAD"].values
    hk_times = nom_hk_scan["PACKET_ICIE_TIME"].values

    # Build per-face OBSID lists from the single source-of-truth mapping
    face_obsids: dict[int, list[int]] = {}
    for obsid, face_id in OBSID_TO_FACE_IDENTIFIER.items():
        face_num = FACE_IDENTIFIER_TO_FACE_NUM[face_id]
        face_obsids.setdefault(face_num, []).append(obsid)

    events: list[SolarCalEvent] = []
    for face_num, obsids_for_face in sorted(face_obsids.items()):
        # Collect individual OBSID run windows for this face
        face_windows: list[tuple[np.datetime64, np.datetime64]] = []
        for obsid in obsids_for_face:
            face_windows.extend(
                l1a_event_utils.detect_value_runs(hk_obsids, hk_times, obsid, min_packets)
            )
        if not face_windows:
            logger.warning("Face %d: no windows detected — skipping.", face_num)
            continue

        face_windows.sort(key=lambda w: w[0])

        # Group consecutive windows into orbital passes
        passes: list[list[tuple[np.datetime64, np.datetime64]]] = [[face_windows[0]]]
        for win in face_windows[1:]:
            if (win[0] - passes[-1][-1][1]) < pass_gap:
                passes[-1].append(win)
            else:
                passes.append([win])

        logger.info("Face %d: %d window(s) → %d pass(es).", face_num, len(face_windows), len(passes))

        for pass_idx, pass_windows in enumerate(passes):
            t_start = min(w[0] for w in pass_windows)
            t_end = max(w[1] for w in pass_windows)
            events.append(
                SolarCalEvent(
                    face_num=face_num,
                    pass_idx=pass_idx,
                    t_event_start=t_start,
                    t_event_end=t_end,
                )
            )

    return events


# ---------------------------------------------------------------------------
# Step 2: Input preparation (writes pre-windowed files for one event)
# ---------------------------------------------------------------------------


def write_event_inputs(
    event: SolarCalEvent,
    data_dir: Path,
    output_dir: Path,
    pad: np.timedelta64 = EVENT_PAD,
) -> tuple[Path, Path, Path] | None:
    """Write the three pre-windowed L1A input files for a single solar-cal event.

    Applies symmetric *pad* around the raw event window, then crops NOM-HK,
    RAD-SAMPLE, and PEV-SW-STAT to that window and writes each with a valid
    Libera filename.  The combiner (``solar_cal_combiner.algorithm``) will
    receive these compact files rather than full 24-hour products.

    Parameters
    ----------
    event : SolarCalEvent
        Detected event describing the face, pass index, and unpadded times.
    data_dir : Path
        Directory containing the source (full 24-hour) L1A files.
    output_dir : Path
        Directory where the three pre-windowed files are written.
    pad : np.timedelta64
        Symmetric time padding applied around the unpadded event window.

    Returns
    -------
    tuple[Path, Path, Path] | None
        ``(nom_hk_path, rad_path, pev_path)`` for the three written files, or
        ``None`` if any product yields zero packets after windowing (event skipped).
    """
    t0 = event.t_event_start - pad
    t1 = event.t_event_end + pad
    label = f"face{event.face_num}_pass{event.pass_idx}"
    pad_minutes = int(pad / np.timedelta64(60, "s"))

    logger.info(
        "Face %d pass %d: event [%s — %s] + %d min padding → window [%s — %s]",
        event.face_num, event.pass_idx,
        event.t_event_start, event.t_event_end,
        pad_minutes, t0, t1,
    )

    nom_hk_path = _write_windowed_product(
        data_dir, output_dir,
        product_token="NOM-HK-DECODED",
        product_identifier=DataProductIdentifier.l1a_icie_nom_hk_decoded,
        t0=t0, t1=t1, label=label,
    )
    if nom_hk_path is None:
        logger.warning(
            "Face %d pass %d: NOM-HK crop produced no packets — skipping event.",
            event.face_num, event.pass_idx,
        )
        return None

    rad_path = _write_windowed_product(
        data_dir, output_dir,
        product_token="RAD-SAMPLE-DECODED",
        product_identifier=DataProductIdentifier.l1a_icie_rad_sample_decoded,
        t0=t0, t1=t1, label=label,
        secondary_time_dim="RAD_SAMPLE_FPE_TIME",
    )
    if rad_path is None:
        logger.warning(
            "Face %d pass %d: no RAD-SAMPLE data in window — skipping event.",
            event.face_num, event.pass_idx,
        )
        return None

    pev_path = _write_windowed_product(
        data_dir, output_dir,
        product_token="PEV-SW-STAT-DECODED",
        product_identifier=DataProductIdentifier.l1a_pev_sw_stat_decoded,
        t0=t0, t1=t1, label=label,
    )
    if pev_path is None:
        logger.warning(
            "Face %d pass %d: no PEV-SW-STAT data in window — skipping event.",
            event.face_num, event.pass_idx,
        )
        return None

    return nom_hk_path, rad_path, pev_path


# ---------------------------------------------------------------------------
# Step 3: Manifest generation (orchestrator)
# ---------------------------------------------------------------------------


def generate_solar_cal_manifests(
    data_dir: Path,
    output_dir: Path,
    pad: np.timedelta64 = EVENT_PAD,
    min_packets: int = MIN_PACKETS,
    pass_gap: np.timedelta64 = PASS_GAP,
) -> list[Path]:
    """Detect solar-cal events and generate one input manifest per face+pass.

    Orchestrates the full pre-processing pipeline:

    1. :func:`detect_solar_cal_events` — fast OBSID scan, no I/O.
    2. :func:`write_event_inputs` — crop and write the three L1A files per event.
    3. Build and write an input manifest ready for
       :func:`~libera_rad.calibration.solar_cal_combiner.algorithm`.

    Parameters
    ----------
    data_dir : Path
        Directory containing full 24-hour L1A ``*.nc`` files for NOM-HK,
        RAD-SAMPLE, and PEV-SW-STAT products.
    output_dir : Path
        Directory where pre-windowed files and manifest files are written.
        Created automatically if it does not exist.
    pad : np.timedelta64
        Symmetric time padding applied around each event window (default 5 min).
    min_packets : int
        Minimum consecutive packets required for a valid OBSID run (default 10).
    pass_gap : np.timedelta64
        Maximum time gap that still counts as the same orbital pass (default 2 h).

    Returns
    -------
    list[Path]
        Paths of all written manifest files, one per detected face+pass event.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    events = detect_solar_cal_events(data_dir, min_packets=min_packets, pass_gap=pass_gap)
    if not events:
        logger.warning("No solar-cal events detected in %s — no manifests written.", data_dir)
        return []

    logger.info("Detected %d event(s). Preparing inputs and manifests ...", len(events))

    manifest_paths: list[Path] = []
    for event in events:
        input_paths = write_event_inputs(event, data_dir, output_dir, pad=pad)
        if input_paths is None:
            continue

        nom_hk_path, rad_path, pev_path = input_paths
        t0 = event.t_event_start - pad
        t1 = event.t_event_end + pad

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
        input_manifest.add_files(nom_hk_path, rad_path, pev_path)
        input_manifest.add_desired_time_range(
            start_datetime=pd.Timestamp(t0).to_pydatetime().replace(tzinfo=UTC),
            end_datetime=pd.Timestamp(t1).to_pydatetime().replace(tzinfo=UTC),
        )
        manifest_path = Path(input_manifest.write(out_path=output_dir))
        manifest_paths.append(manifest_path)
        logger.info(
            "Face %d pass %d: manifest written → %s",
            event.face_num, event.pass_idx, manifest_path,
        )

    return manifest_paths


# ---------------------------------------------------------------------------
# Internal helper — write one pre-windowed product file
# ---------------------------------------------------------------------------


def _write_windowed_product(
    data_dir: Path,
    output_dir: Path,
    product_token: str,
    product_identifier: DataProductIdentifier,
    t0: np.datetime64,
    t1: np.datetime64,
    label: str,
    secondary_time_dim: str | None = None,
) -> Path | None:
    """Filter, time-slice, and write one L1A product to a single output file.

    Selects source files from *data_dir* whose filename time range overlaps
    ``[t0, t1]``, opens and concatenates them, slices to ``[t0, t1]`` via
    :func:`~libera_rad.calibration.l1a_event_utils.slice_dataset_to_time_window`,
    clears stale NetCDF encodings, and writes with a valid Libera filename.

    Parameters
    ----------
    data_dir : Path
        Source directory containing L1A ``*.nc`` files.
    output_dir : Path
        Destination directory for the written file.
    product_token : str
        Filename substring identifying the product type
        (e.g. ``"NOM-HK-DECODED"``).
    product_identifier : DataProductIdentifier
        Used to construct the output Libera filename.
    t0, t1 : np.datetime64
        Time window (inclusive) for both file selection and data slicing.
    label : str
        Short label for log messages (e.g. ``"face1_pass0"``).
    secondary_time_dim : str | None
        Optional independent secondary time dimension to also slice
        (e.g. ``"RAD_SAMPLE_FPE_TIME"``).

    Returns
    -------
    Path or None
        Path to the written file, or ``None`` if no data falls in ``[t0, t1]``.
    """
    all_files = sorted(data_dir.glob(f"*{product_token}*.nc"))
    if not all_files:
        logger.warning("%s: no files found in %s", product_token, data_dir)
        return None

    selected = l1a_event_utils.filter_files_by_time_window(
        all_files, [(t0, t1)], pad=np.timedelta64(0, "s")
    )
    if not selected:
        logger.warning("%s (%s): no files overlap window [%s — %s]", product_token, label, t0, t1)
        return None

    logger.info("%s (%s): loading %d / %d file(s) ...", product_token, label, len(selected), len(all_files))
    ds = l1a_event_utils.open_and_sort_l1a_files(selected)

    ds = l1a_event_utils.slice_dataset_to_time_window(
        ds, t0, t1, secondary_time_dim=secondary_time_dim
    )

    n_packets = ds.sizes.get("PACKET", 0)
    if n_packets == 0:
        logger.warning("%s (%s): 0 packets after slicing — skipping.", product_token, label)
        return None

    if secondary_time_dim:
        n_secondary = ds.sizes.get(secondary_time_dim, 0)
        logger.info(
            "%s (%s): %d packets, %d %s samples retained.",
            product_token, label, n_packets, n_secondary, secondary_time_dim,
        )
    else:
        logger.info("%s (%s): %d packets retained.", product_token, label, n_packets)

    actual_times = ds["PACKET_ICIE_TIME"].values
    utc_start = pd.Timestamp(actual_times.min()).to_pydatetime().replace(tzinfo=UTC)
    utc_end = pd.Timestamp(actual_times.max()).to_pydatetime().replace(tzinfo=UTC)

    libera_filename = LiberaDataProductFilename.from_filename_parts(
        product_name=product_identifier,
        version=f"V{version().replace('.', '-')}",
        utc_start=utc_start,
        utc_end=utc_end,
        basepath=output_dir,
    )
    out_path = Path(libera_filename.path)

    # Clear stale source-file encodings to prevent NetCDF write warnings
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].encoding.clear()

    ds.to_netcdf(out_path)
    logger.info("%s (%s): written → %s", product_token, label, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Script entry point — IOV test data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Directory containing full 24-hour L1A files for NOM-HK, RAD-SAMPLE,
    # and PEV-SW-STAT.  Update to point at your IOV test data.
    L1A_DIR = Path("~/Documents/Libera/test_data/IOV/IOV_l1a").expanduser()

    # Working/output directory — pre-windowed files, manifests, and solar-cal
    # combined output products are all written here.
    WORK_DIR = Path("~/Desktop/solar_cal_files").expanduser()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["PROCESSING_PATH"] = str(WORK_DIR)

    logger.info("=== Solar Cal Runner ===")
    logger.info("L1A directory : %s", L1A_DIR)
    logger.info("Working dir   : %s", WORK_DIR)

    # Step 1: Detect events, write pre-windowed files, build manifests
    manifests = generate_solar_cal_manifests(data_dir=L1A_DIR, output_dir=WORK_DIR)

    if not manifests:
        logger.error("No solar-cal events detected. Check L1A_DIR: %s", L1A_DIR)
    else:
        logger.info("Generated %d manifest(s). Running combiner ...", len(manifests))

        # Step 2: Run the combiner algorithm for each manifest
        for mf in manifests:
            logger.info("--- Processing manifest: %s ---", mf)
            output_manifest = algorithm(mf)
            logger.info("Output manifest: %s", output_manifest)

    logger.info("=== Done ===")

