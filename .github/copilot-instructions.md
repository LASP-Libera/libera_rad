`libera_rad` is the LASP Libera radiometer L1B algorithm package: it converts L1A radiometer
digital numbers to calibrated radiances and writes NetCDF4 L1B output products via a
seven-step pipeline. It is written in Python (>=3.11) and managed with Poetry.

Detailed coding rules (standards, testing conventions, key patterns, and agent restrictions)
are in [.github/instructions/libera_rad.instructions.md](.github/instructions/libera_rad.instructions.md),
which Copilot applies automatically to all files via its `applyTo: "**"` front matter.
