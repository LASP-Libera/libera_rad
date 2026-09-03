"""Constants for L1B geolocation and geometry.

Collects the SPICE frames and science tunables the geometry code depends on, so they are
named and reviewable in one place rather than embedded as defaults at their call sites.
"""

import numpy as np

# --- SPICE observer frames ------------------------------------------------------------

# Spacecraft frames the Libera FK defines. Libera flies on JPSS-4; NOAA-20 is the alternate
# bus configuration carried in the same kernel set.
SPACECRAFT_OBSERVERS: tuple[str, ...] = ("JPSS4_SC", "NOAA20_SC")
DEFAULT_SPACECRAFT_OBSERVER = "JPSS4_SC"

# The single radiometer frame the Libera FK defines. The four radiometer channels (SW, SSW,
# LW, TOT) are co-boresighted and share the ``LIBERA_RAD`` frame, so one geometry query
# serves all four. Should the FK ever carry real per-channel boresight offsets, per-channel
# frames would return here, each channel would need its own query, and the product would
# need per-channel geometry fields.
INSTRUMENT_OBSERVERS: tuple[str, ...] = ("LIBERA_RAD",)
DEFAULT_INSTRUMENT_OBSERVER = "LIBERA_RAD"

# --- Science tunables -----------------------------------------------------------------

# Off-nadir gate for Clock_Angle_Rate. The clock angle is an azimuth about nadir, so its rate
# is singular there; as the cross-track scan crosses nadir the azimuth swings ~180 degrees
# faster than the 100 Hz sampling resolves, making the finite difference an aliasing artifact
# rather than a derivative. Ungated, samples reach ~8000 deg/s against a declared valid_range
# of [-20, 20].
#
# The largest surviving rate sits at the gate boundary and falls off as 1/gate**2: for a scan of
# angular speed w whose closest approach to nadir is b, the rate at cone angle g is w*b/g**2.
# Measured on the in-repo granule at the 100 Hz product cadence across 65 minutes of kernel
# coverage, that envelope is 43.4 * (6/g)**2 deg/s, so satisfying the declared range needs
# g >= 8.8 degrees. 12 degrees holds the surviving rate near 10.8 deg/s -- about a factor of two
# of margin against a faster scan or a wider nadir miss -- and fills 15% of samples where a
# 6 degree gate filled 7.5%.
#
# TODO[LIBSDC-739]: placeholder pending science confirmation of the useful off-nadir limit. The
# measurement above covers one cross-track granule; other scan modes change both w and b.
CLOCK_RATE_MIN_CONE_ANGLE_DEG = np.float32(12.0)
