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

# Angular slack added around the radiometer field of view when deciding whether the Moon's
# light reaches the detectors. The geometric FOV itself is read from the instrument kernel
# rather than hardcoded, and the lunar disk's own apparent radius (about 0.24 - 0.29 deg seen from orbit, from
# curryer's ``moon_angular_radius``) is added separately, so this constant carries only the
# margin beyond "any part of the lunar disk is inside the geometric cone".
#
# 5 degrees is roughly five times the IK's current 1 deg half-angle. It covers the three
# things that put lunar signal on the detectors outside the geometric cone: near-field stray
# light, which does not stop at the FOV edge and is the governing concern for an off-limb dark
# measurement; pointing-knowledge and CK-interpolation error (<~0.1 deg); and the lunar disk
# itself if the angular-radius term is ever dropped. It stays small against the Moon's
# ~0.5 deg/hr apparent motion, so a flagged interval is set by the scan sweeping past the Moon
# rather than by the buffer smearing it out.
#
# TODO[LIBSDC-827]: placeholder pending science sign-off. Note the geometric FOV it widens is
# itself provisional -- the IK carries TODO[LIBSDC-601] for the ground-measured value -- so
# this number should be revisited when the real FOV lands.
MOON_FOV_BUFFER_DEG = np.float32(5.0)

# Product fill for the integer Moon-in-view flag, distinguishing "no pointing data" from
# "Moon not in view".
MOON_IN_FOV_FILL_VALUE = np.int8(-128)

# --- Gimbal scan-rate conventions ------------------------------------------------------

# L1B output cadence; the scan rates are differenced over this grid.
L1B_OUTPUT_HZ = 100

# Scan-rate intervals further than the tolerance from nominal fill both rates. Rates divide by
# the measured interval; heritage divides by the nominal period, which at this tolerance could
# misreport a rate by up to a factor of two.
NOMINAL_SAMPLE_PERIOD_S = 1.0 / L1B_OUTPUT_HZ
SAMPLE_PERIOD_TOLERANCE_S = 0.005

# Elevation encoder reading at nadir. Heritage puts nadir at 90 degrees; the Libera encoder
# reads 0 at nadir and sweeps +/-72 degrees about it. Inferred from the heritage sign rules,
# not documented.
GIMBAL_NADIR_ELEVATION_DEG = np.float32(0.0)
