# Version Changes

## 0.5.6

- Production geolocation: populate `Subsatellite_*` via `LIBERA_BASE` nadir (reusing jpss_only subsatellite function), motor `Azimuth`/`Elevation` from CK frames, and derive `Colatitude` from instrument latitude.

## 0.5.5

- Add `jpss_only` manifest configuration: load only JPSS-SPK and JPSS-CK dynamic kernels, compute subsatellite geolocation via `LIBERA_BASE` nadir ellipsoid intersection (no motor CK), populate instrument and `Subsatellite_*` lat/lon from a single call, and write Azimuth/Elevation as 0°. Warn when other SPICE files are listed but skipped.
- Refactor SPICE manifest handling: validate required kernel product IDs, reject duplicates, and return kernels in furnish order.
- Extend `use_geo: false` mode: Azimuth and Elevation use product fill (`-999`); warn when any `.bc`/`.bsp` files are listed but skipped.
- Manifest flag `jpss_only` requires a truthy value; `use_geo: false` and `jpss_only: true` together raise `ValueError`.
- L1A reads use `decode_times=True`; pipeline timestamps are `datetime64[ns]` only (removed `_CCSDS_EPOCH` numeric conversion paths).
- Populate `Operational_Mode` from packet OBSID; update product definition dtypes and radiance valid ranges; set `algorithm_version` at runtime.

## 0.5.4

- Add optional `use_geo` manifest configuration for ground-calibration processing. Omitting `use_geo` or setting it to `true` runs SPICE geolocation as in production. Only `use_geo: false` skips `.bc`/`.bsp` during manifest read, bypasses `KernelManager`, and writes standard product fill values for latitude, longitude, and altitude. Output manifest configuration is propagated from the input manifest.

## 0.5.3

- Use libera_utils `KernelFileCache`/`KernelManager.load_libera_dynamic_kernels` for dynamic kernel materialization and add an integration assertion that dynamic kernel loads populate the user cache. Callers must pass a **sequence** of kernel paths; integration tests and learning notebooks build an explicit sorted path list from test data directories.

## 0.5.2

- Correct the downsampling method in gain calibration so that no additional filtering occurs after calibration.

## 0.5.1

- Update the L1B Radiometer 4-channel product definition to be compatible with the latest version of libera_utils with dimension and type enforcement

## 0.5.0

- Initial l1b production algorithm, including gain calibration, geolocation, and radiance calculation.

## 0.4.5

- Adding the first geolocation calculation. Uses KernelManager from libera_utils and curryer tools.

## 0.4.4

- Update the format of the product definition for the L1B Radiometer 4-channel product to be usable with breaking changes made in libera_utils 5.0.0

## 0.4.3

- Add gain_calibration module and standard transfer function.

## 0.4.2

- Add baseline product definition for the L1B Radiometer 4-channel product, which will
  serve as the ERB continuity product during Y1 of Libera operations.

## 0.4.1

- Update dependencies to latest version of libera_utils (4.0.0)
- First set of clean up to cli and algorithm code

## 0.4.0

- Standardize the repo with pre-commit and formatting tools to match SDC standards

## 0.3.0

- Created the alpha version of the l1b radiometer algorithm
- Added learning notebooks to explain the science behind the algorithm
- Added ground calibration constants files from instrument team
- Added unit tests for the l1b radiometer algorithm to test against instrument team reference data

## 0.2.0

- Add `libera-rad` CLI for running L1b Radiometer processing from inside a Docker image
- Add "pass-through" algorithm that can be used to test processing orchestration for
  spice-jpss, spice-azel, l1b-rad, and l1b-cam processing steps
