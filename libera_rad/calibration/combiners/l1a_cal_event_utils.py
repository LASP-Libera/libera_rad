"""L1A calibration event utilities — filenames, loading, and time windows.

Shared helpers for calibration combiners that operate on decoded L1A NetCDF
products:

- Parse Libera filename time ranges
- Filter file lists by overlapping time windows (default 60 s pad)
- Open, validate, concatenate, and sort files by ``PACKET_ICIE_TIME``
- Detect contiguous runs of an integer telemetry field (e.g. OBSID)
- Scan directories for event windows from a single field variable
- Slice datasets to inclusive ``[t0, t1]`` on packet time (and optional FPE time)
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from libera_utils import Manifest
from libera_utils.io.filenaming import LiberaDataProductFilename

from libera_rad.l1b import read_all_input_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-level utilities
# ---------------------------------------------------------------------------


def parse_libera_filename_times(path: Path) -> tuple[np.datetime64, np.datetime64] | None:
    """Extract (t_start, t_end) embedded in a Libera L1A filename.

    Delegates to :class:`~libera_utils.io.filenaming.LiberaDataProductFilename`
    so parsing is consistent with the canonical Libera filename convention.

    Parameters
    ----------
    path : Path
        Path whose filename follows the Libera data product naming convention.

    Returns
    -------
    tuple[np.datetime64, np.datetime64] | None
        ``(t_start, t_end)`` with second resolution, or ``None`` when the
        filename does not match the Libera data product regex.
    """
    try:
        fn = LiberaDataProductFilename.from_file_path(str(path))
        parts = fn.filename_parts
        t_start = np.datetime64(parts.utc_start.replace(tzinfo=None), "s")
        t_end = np.datetime64(parts.utc_end.replace(tzinfo=None), "s")
        return t_start, t_end
    except (ValueError, AttributeError):
        return None


def filter_files_by_time_window(
    all_files: list[Path],
    windows: list[tuple[np.datetime64, np.datetime64]],
    pad: np.timedelta64 = np.timedelta64(60, "s"),
) -> list[Path]:
    """Return only those files whose filename time range overlaps any window.

    Parameters
    ----------
    all_files : list[Path]
        Candidate file paths (pre-sorted if ordering matters).
    windows : list[tuple[np.datetime64, np.datetime64]]
        ``(t_start, t_end)`` windows to test against.
    pad : np.timedelta64
        Extra time buffer applied symmetrically around each window (default 60 s).
        Files are included if their range overlaps the padded window.

    Returns
    -------
    list[Path]
        Subset of *all_files* that overlap at least one window.
    """
    selected: list[Path] = []
    for f in all_files:
        ft = parse_libera_filename_times(f)
        if ft is None:
            # Cannot parse — include conservatively
            selected.append(f)
            continue
        f_start, f_end = ft
        for w_start, w_end in windows:
            if f_end >= (w_start - pad) and f_start <= (w_end + pad):
                selected.append(f)
                break
    return selected


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def open_and_sort_l1a_files(files: list[Path]) -> xr.Dataset:
    """Open, concatenate, and sort L1A NetCDF files by ``PACKET_ICIE_TIME``.

    For products with a single ``PACKET`` dimension this uses
    ``xr.open_mfdataset``.  For RAD-SAMPLE, which has two independent
    dimensions (``PACKET`` and ``RAD_SAMPLE_FPE_TIME``) that vary in size
    between files, arrays are concatenated manually to avoid xarray alignment
    errors.

    Parameters
    ----------
    files : list[Path]
        One or more NetCDF files for the same product type.

    Returns
    -------
    xr.Dataset
        Concatenated dataset, time-sorted along the ``PACKET`` dimension.

    Raises
    ------
    ValueError
        If input files do not share the same Libera ``data_product_id``.
    """
    if len(files) > 1:
        product_ids = [LiberaDataProductFilename.from_file_path(str(path)).data_product_id for path in files]
        if len(set(product_ids)) > 1:
            raise ValueError(f"All input files must be the same L1A product type; found: {sorted(set(product_ids))}")

    if len(files) == 1:
        ds = xr.open_dataset(files[0]).load()
        if "PACKET_ICIE_TIME" in ds.coords:
            ds = ds.isel(PACKET=np.argsort(ds["PACKET_ICIE_TIME"].values))
        return ds

    with xr.open_dataset(files[0]) as _peek:
        has_fpe_dim = "RAD_SAMPLE_FPE_TIME" in _peek.dims

    if not has_fpe_dim:
        parts = [xr.open_dataset(f).load() for f in files]
        ds = xr.concat(parts, dim="PACKET")
        if "PACKET_ICIE_TIME" in ds.coords:
            ds = ds.isel(PACKET=np.argsort(ds["PACKET_ICIE_TIME"].values))
        return ds

    # RAD-SAMPLE: two independent dimensions with varying lengths per file.
    # Concatenate numpy arrays directly to avoid xarray broadcasting issues.
    packet_times_list: list[np.ndarray] = []
    fpe_times_list: list[np.ndarray] = []
    data_vars_arrays: dict[str, list[np.ndarray]] = {}

    for f in files:
        with xr.open_dataset(f) as ds_f:
            ds_f.load()
            packet_times_list.append(ds_f["PACKET_ICIE_TIME"].values)
            fpe_times_list.append(ds_f["RAD_SAMPLE_FPE_TIME"].values)
            for var in ds_f.data_vars:
                data_vars_arrays.setdefault(var, []).append(ds_f[var].values)
            for coord in ds_f.coords:
                if coord not in ("PACKET_ICIE_TIME", "RAD_SAMPLE_FPE_TIME"):
                    data_vars_arrays.setdefault(coord, []).append(ds_f[coord].values)

    packet_times = np.concatenate(packet_times_list)
    fpe_times = np.concatenate(fpe_times_list)
    pkt_order = np.argsort(packet_times)
    packet_times = packet_times[pkt_order]

    n_packets = len(packet_times)
    n_fpe = len(fpe_times)

    coords: dict = {
        "PACKET_ICIE_TIME": ("PACKET", packet_times),
        "RAD_SAMPLE_FPE_TIME": ("RAD_SAMPLE_FPE_TIME", fpe_times),
    }
    data: dict = {}
    for var, arrays in data_vars_arrays.items():
        arr = np.concatenate(arrays)
        if arr.ndim == 1:
            if arr.shape[0] == n_packets:
                data[var] = ("PACKET", arr[pkt_order])
            elif arr.shape[0] == n_fpe:
                data[var] = ("RAD_SAMPLE_FPE_TIME", arr)
            else:
                logger.warning(
                    "Variable %s has unexpected length %d (n_packets=%d, n_fpe=%d); assigning no dim.",
                    var,
                    arr.shape[0],
                    n_packets,
                    n_fpe,
                )
                data[var] = arr
        elif arr.ndim == 2 and arr.shape[0] == n_packets:
            data[var] = (["PACKET", "RAD_SAMPLE_FPE_TIME"], arr[pkt_order])
        else:
            logger.warning("Variable %s has unexpected shape %s; assigning no dim.", var, arr.shape)
            data[var] = arr

    return xr.Dataset(data, coords=coords)


def load_l1a_product(
    data_dir: Path,
    product_name: str,
    filter_windows: list[tuple[np.datetime64, np.datetime64]] | None = None,
    pad: np.timedelta64 = np.timedelta64(60, "s"),
) -> xr.Dataset:
    """Open and concatenate L1A files for a product type from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory to scan for ``*.nc`` files.
    product_name : str
        Filename substring identifying the product type
        (e.g. ``"NOM-HK-DECODED"``).
    filter_windows : list[tuple[np.datetime64, np.datetime64]] | None
        When provided, only files whose filename time range overlaps at least
        one window (within *pad*) are opened.
    pad : np.timedelta64
        Extra time buffer applied around each window for file selection
        (default 60 s).

    Returns
    -------
    xr.Dataset
        Concatenated and time-sorted dataset.

    Raises
    ------
    FileNotFoundError
        If no matching files are found in *data_dir*.
    """
    all_files = sorted(data_dir.glob(f"*{product_name}*.nc"))
    if not all_files:
        raise FileNotFoundError(f"No '*{product_name}*.nc' files in {data_dir}")

    if filter_windows is not None:
        files = filter_files_by_time_window(all_files, filter_windows, pad)
        logger.info("%s: %d / %d file(s) selected by window filter.", product_name, len(files), len(all_files))
    else:
        files = all_files
        logger.info("%s: loading all %d file(s).", product_name, len(all_files))

    if not files:
        raise FileNotFoundError(f"No '{product_name}' files overlap the requested windows.")

    return open_and_sort_l1a_files(files)


# ---------------------------------------------------------------------------
# Run / event-window detection
# ---------------------------------------------------------------------------


def detect_value_runs(
    field_values: np.ndarray,
    times: np.ndarray,
    target_value: int,
    min_count: int = 10,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Return ``(t_start, t_end)`` pairs for every contiguous run of *target_value*.

    Generic over any integer telemetry field — not just ``ICIE__SW_OBSID_RAD``.

    Parameters
    ----------
    field_values : np.ndarray
        Integer array aligned with *times*.
    times : np.ndarray
        Corresponding ``np.datetime64`` timestamp array.
    target_value : int
        Field value that constitutes the event.
    min_count : int
        Minimum run length required to keep a window (default 10).

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        ``(t_start, t_end)`` pairs for every qualifying contiguous run.
    """
    mask = field_values == target_value
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(times[s], times[e - 1]) for s, e in zip(starts, ends) if (e - s) >= min_count]


def scan_files_for_field_windows(
    data_dir: Path,
    product_name: str,
    target_values: list[int],
    field_var: str = "ICIE__SW_OBSID_RAD",
    time_var: str = "PACKET_ICIE_TIME",
    min_count: int = 10,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Scan L1A files reading only one field variable to detect event windows.

    Much faster than loading the full product; intended for initial event
    detection before loading any science data.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*.nc`` L1A files.
    product_name : str
        Filename substring identifying the product (e.g. ``"NOM-HK-DECODED"``).
    target_values : list[int]
        Field values that indicate an event of interest.
    field_var : str
        Variable name to scan (default ``"ICIE__SW_OBSID_RAD"``).
    time_var : str
        Timestamp coordinate name (default ``"PACKET_ICIE_TIME"``).
    min_count : int
        Minimum run length to keep a window (default 10).

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        Sorted list of ``(t_start, t_end)`` windows across all target values.

    Raises
    ------
    FileNotFoundError
        If no matching files exist in *data_dir*.
    """
    files = sorted(data_dir.glob(f"*{product_name}*.nc"))
    if not files:
        raise FileNotFoundError(f"No '*{product_name}*.nc' files in {data_dir}")

    logger.info(
        "Scanning %d %s file(s) for %s values %s ...",
        len(files),
        product_name,
        field_var,
        target_values,
    )

    all_field_values: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if field_var not in ds or time_var not in ds:
                continue
            all_field_values.append(ds[field_var].values)
            all_times.append(ds[time_var].values)

    if not all_field_values:
        return []

    field_arr = np.concatenate(all_field_values)
    times_arr = np.concatenate(all_times)
    order = np.argsort(times_arr)
    field_arr, times_arr = field_arr[order], times_arr[order]

    windows: list[tuple[np.datetime64, np.datetime64]] = []
    for target in target_values:
        windows.extend(detect_value_runs(field_arr, times_arr, target, min_count))

    windows.sort(key=lambda w: w[0])
    return windows


# ---------------------------------------------------------------------------
# Time-window slicing
# ---------------------------------------------------------------------------


def slice_dataset_to_time_window(
    ds: xr.Dataset,
    t0: np.datetime64,
    t1: np.datetime64,
    packet_time_var: str = "PACKET_ICIE_TIME",
    secondary_time_dim: str | None = None,
) -> xr.Dataset:
    """Slice a decoded L1A Dataset to packets (and optionally samples) in ``[t0, t1]``.

    Applies an inclusive mask on *packet_time_var* along ``PACKET``. When
    *secondary_time_dim* is set, that dimension is sliced independently (e.g.
    ``RAD_SAMPLE_FPE_TIME`` on RAD-SAMPLE products).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``PACKET`` dimension indexed by *packet_time_var*.
    t0 : np.datetime64
        Window start time (inclusive).
    t1 : np.datetime64
        Window end time (inclusive).
    packet_time_var : str
        Name of the packet-level time coordinate (default ``"PACKET_ICIE_TIME"``).
    secondary_time_dim : str | None
        Name of an independent secondary time dimension to also slice (e.g.
        ``"RAD_SAMPLE_FPE_TIME"``).  When provided the Dataset is sliced along
        both ``PACKET`` and this dimension independently.

    Returns
    -------
    xr.Dataset
        Time-sliced dataset.
    """
    pkt_times = ds[packet_time_var].values
    pkt_mask = (pkt_times >= t0) & (pkt_times <= t1)
    sel: dict[str, np.ndarray] = {"PACKET": pkt_mask}

    if secondary_time_dim is not None and secondary_time_dim in ds.dims:
        sec_times = ds[secondary_time_dim].values
        sec_mask = (sec_times >= t0) & (sec_times <= t1)
        sel[secondary_time_dim] = sec_mask
        logger.debug(
            "slice_dataset_to_time_window [%s — %s]: %d packets, %d %s samples.",
            t0,
            t1,
            int(pkt_mask.sum()),
            int(sec_mask.sum()),
            secondary_time_dim,
        )
    else:
        logger.debug(
            "slice_dataset_to_time_window [%s — %s]: %d packets.",
            t0,
            t1,
            int(pkt_mask.sum()),
        )

    return ds.isel(sel)


def read_calibration_manifest_data(input_manifest: Manifest) -> dict[str, xr.Dataset]:
    """Load decoded L1A datasets from a calibration combiner input manifest.

    Calibration combiners do not use SPICE/geolocation. Input manifests therefore
    default to ``use_geo: false`` so :func:`~libera_rad.l1b.read_all_input_data`
    does not require SPICE kernel products.
    """
    if input_manifest.configuration.get("use_geo", True):
        read_manifest = input_manifest.model_copy(
            update={"configuration": {**input_manifest.configuration, "use_geo": False}}
        )
    else:
        read_manifest = input_manifest
    all_data, _ = read_all_input_data(read_manifest)
    return all_data
