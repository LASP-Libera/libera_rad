import logging

import xarray as xr
from libera_utils.constants import LiberaApid

from libera_rad.calibration.constants import COMBINER_CCSDS_DROP_FIELDS, COMBINER_CCSDS_KEEP_FIELDS

logger = logging.getLogger(__name__)


def merge_l1a_decoded_datasets(
    datasets: list[xr.Dataset],
    ccsds_header_keep_fields: list[str] = COMBINER_CCSDS_KEEP_FIELDS,
    ccsds_header_drop_fields: list[str] = COMBINER_CCSDS_DROP_FIELDS,
) -> xr.Dataset:
    """
    Takes a list of L1A decoded Xarray Datasets and merges them into a single
    flat dataset with the CCSDS header variables kept and dropped as specified.

    Parameters
    ----------
    datasets : list[xr.Dataset]
        List of L1A decoded Xarray Datasets to merge.
    ccsds_header_keep_fields : list[str], optional
        List of CCSDS header fields to keep.
    ccsds_header_drop_fields : list[str], optional
        List of CCSDS header fields to drop.
    Returns
    -------
    xr.Dataset
        Merged L1A decoded Xarray Dataset with the CCSDS header variables kept and dropped as specified.
    """
    prepared_datasets = []

    for ds in datasets:
        # Identify the dataset's APID to determine its prefix
        # We grab the APID from the first packet.
        if "PKT_APID" not in ds:
            raise ValueError("Dataset does not contain a PKT_APID variable")

        libera_apid = LiberaApid(ds["PKT_APID"].isel(PACKET=0).values)
        prefix = str(libera_apid.data_product_id).split("-")[:-1]
        prefix = "_".join(prefix)

        # Create a copy to avoid mutating the inputs directly
        ds_prep = ds.copy()

        # Clear stale NetCDF encoding metadata (chunksizes, fletcher32, source, etc.)
        # carried over from the source file; downstream product_definition enforcement
        # will warn/fail on any encoding keys it does not expect.
        for var in ds_prep.variables.values():
            var.encoding.clear()

        # Remove the CCSDS header variables in the drop list (only those present)
        vars_to_drop = [v for v in ccsds_header_drop_fields if v in ds_prep]
        missing_drop_vars = [v for v in ccsds_header_drop_fields if v not in ds_prep]
        if missing_drop_vars:
            logger.debug("Variables not found in dataset and will not be dropped: %s", missing_drop_vars)
        ds_prep = ds_prep.drop_vars(vars_to_drop)

        # Prefix them so they don't collide (e.g., RAD_SAMPLE_SRC_SEQ_CTR)
        rename_dict = {v: f"{prefix}_{v}" for v in ccsds_header_keep_fields if v in ds_prep}
        missing_prefix_vars = [v for v in ccsds_header_keep_fields if v not in ds_prep]
        if rename_dict:
            logger.debug("Variables prefixed with %r: %s", prefix, rename_dict)
        if missing_prefix_vars:
            logger.debug("CCSDS keep fields not found in dataset and will not be prefixed: %s", missing_prefix_vars)
        ds_prep = ds_prep.rename_vars(rename_dict)

        # Each decoded product uses the same dimension name "PACKET" for its CCSDS axis.
        # xr.merge would align those on one shared PACKET dimension; rename so each stream
        # keeps its own axis length and indexing independent of the others.
        if "PACKET" in ds_prep.dims:
            ds_prep = ds_prep.rename_dims({"PACKET": f"{prefix}_PACKET"})

        prepared_datasets.append(ds_prep)

    # Merge into a single overarching dataset
    merged_ds = xr.merge(prepared_datasets, compat="no_conflicts")

    return merged_ds
