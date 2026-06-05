"""Unit tests for surface geometry angle helpers (optional SPICE integration)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_rad import geolocation

JPSS_KERNEL_DIR = Path(__file__).resolve().parents[1] / "test_data" / "l1b_integration_data"


@pytest.mark.integration
def test_calculate_surface_geometry_angles_jpss_kernels():
    """Surface angles at LIBERA_BASE nadir should be physically plausible."""
    sources = [
        str(JPSS_KERNEL_DIR / "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc"),
        str(JPSS_KERNEL_DIR / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp"),
    ]
    times = pd.date_range("2025-11-20 18:00:00", periods=5, freq="1s").to_numpy(dtype="datetime64[ns]")

    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        lla = geolocation.calculate_libera_base_subsatellite_geolocation(km, times)
        angles = geolocation.calculate_surface_geometry_angles(km, times, lla, spacecraft_body="LIBERA_BASE")

    assert angles["solar_zenith"].shape == (5,)
    assert np.all((angles["solar_zenith"] >= 0) & (angles["solar_zenith"] <= 180))
    assert np.all((angles["viewing_zenith"] >= 0) & (angles["viewing_zenith"] < 1.0))
    assert np.all((angles["relative_azimuth"] >= 0) & (angles["relative_azimuth"] < 360))
