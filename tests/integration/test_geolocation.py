"""Integration tests for geolocation using static and geolocation kernels."""

from datetime import datetime

import numpy as np
import pandas as pd
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad.geolocation import calculate_lat_lon_altitude


def test_geolocation_from_kernels(test_dynamic_kernels_path):
    """
    Test geolocation computation using static and geolocation kernels from the Libera SDC tier 1 geolocation test.
    """
    start_time = datetime(2028, 1, 2, 0, 13, 47)
    end_time = datetime(2028, 1, 2, 0, 29, 12)

    km = KernelManager()
    km.load_libera_dynamic_kernels(test_dynamic_kernels_path, needs_naif_kernels=True, needs_static_kernels=True)

    print(f"Calculating lat/lon/alt from {start_time} to {end_time}")
    time_range = pd.date_range(start_time, end_time, freq="10ms", inclusive="left")

    lat_lon_alt = calculate_lat_lon_altitude(km, time_range)

    print("Mean Lat: ", np.nanmean(lat_lon_alt["lat"]))
    print("Mean Lon: ", np.nanmean(lat_lon_alt["lon"]))

    clon, clat = 23.39, 28.55
    hist, xedge, yedge = np.histogram2d(
        lat_lon_alt["lon"],
        lat_lon_alt["lat"],
        bins=(13, 13),
        range=[[clon - 3, clon + 3], [clat - 3, clat + 3]],
    )
    idx = np.where(hist == hist.max())
    ix, iy = idx[0][0], idx[1][0]

    print(f"Expected focus point:    lon=[{clon}],          lat=[{clat}]")
    print(
        f"2D histogram max between lon=[{xedge[ix]:.3f}, {xedge[ix + 1]:.3f}],"
        f" lat=[{yedge[iy]:.3f}, {yedge[iy + 1]:.3f}]"
    )
    assert ix == hist.shape[0] // 2, ix
    assert iy == hist.shape[0] // 2, iy


def test_dynamic_kernels_materialize_into_cache(monkeypatch, tmp_path, test_dynamic_kernels_path):
    """Dynamic kernels should be materialized into the user cache via KernelFileCache."""
    monkeypatch.setattr(spice_utils.caching, "get_local_cache_dir", lambda: tmp_path)

    kernel_files = sorted(
        [p for p in test_dynamic_kernels_path.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"}],
        key=lambda p: p.name,
    )
    assert kernel_files, f"No dynamic kernels found under {test_dynamic_kernels_path}"
    sources = kernel_files[:2]

    km = KernelManager(cache_timeout_days=7)
    km.load_libera_dynamic_kernels(test_dynamic_kernels_path, needs_naif_kernels=True, needs_static_kernels=True)
    km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)

    for src in sources:
        assert (tmp_path / src.name).is_file(), f"Expected cached kernel missing: {src.name}"
