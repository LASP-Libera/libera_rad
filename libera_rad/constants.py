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

# The four radiometer channel frames the Libera FK defines. All four are currently identity
# rotations relative to ``LIBERA_EL_COORD`` -- the channels are co-boresighted, so geometry
# computed for any one of them is valid for all four, and one query suffices. There is no
# generic ``LIBERA_RAD`` frame to name instead. Should the FK ever carry real per-channel
# boresight offsets, each channel would need its own query and the product would need
# per-channel geometry fields.
INSTRUMENT_OBSERVERS: tuple[str, ...] = ("LIBERA_SW_RAD", "LIBERA_LW_RAD", "LIBERA_TOT_RAD", "LIBERA_SSW_RAD")
DEFAULT_INSTRUMENT_OBSERVER = "LIBERA_SW_RAD"

# --- Science tunables -----------------------------------------------------------------

# Off-nadir gate for Clock_Angle_Rate. The clock angle is an azimuth about nadir, so its rate
# is singular there; as the cross-track scan crosses nadir the azimuth swings ~180 degrees
# faster than the 100 Hz sampling resolves, making the finite difference an aliasing artifact
# rather than a derivative. Ungated, samples reach 7588 deg/s against a declared valid_range
# of [-20, 20]; that range corresponds almost exactly to excluding a 5.30 degree cone, and 6
# degrees adds margin.
#
# TODO[LIBSDC-739]: placeholder pending science confirmation of the useful off-nadir limit.
CLOCK_RATE_MIN_CONE_ANGLE_DEG = np.float32(6.0)
