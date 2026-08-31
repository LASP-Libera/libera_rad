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

# --- Gimbal scan-rate conventions ------------------------------------------------------

# Sample period the scan rates are differenced over, and the tolerance outside which a
# sample spacing is not a usable rate interval. Following the heritage implementation, a
# spacing further than the tolerance from nominal fills both rates rather than dividing by
# a spacing the instrument did not have.
NOMINAL_SAMPLE_PERIOD_S = 0.01
SAMPLE_PERIOD_TOLERANCE_S = 0.005

# Elevation encoder reading that puts the boresight at nadir. The heritage implementation
# writes its cone-rate sign rules around a 90 degree nadir; the Libera elevation encoder
# reads 0 there, so the rules are expressed against this reference rather than a literal 90.
GIMBAL_NADIR_ELEVATION_DEG = np.float32(0.0)
