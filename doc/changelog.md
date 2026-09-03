# Version Changes

## 0.7.0

- Replace type-specific calibration combiners and `*-COMBINED` products with a single ObsID-dispatched `cal-combine` algorithm, selected by the `LIBERA_CAL_OBSID` environment variable. Invoke via `libera-rad cal-combine <manifest>` (docker-compose `cal-combine` service included). Outputs are ObsID-specific: `GAIN`, `NOISE`, `SWC-{λ}NM`, `LWC-{310,320,335,300,305}K`, `SOLAR-{ch}-{PRI,SEC,TER}`.
- Shared gain/noise/swc/lwc/solar event pipeline — NOM-HK ObsID check against `LIBERA_CAL_OBSID`, crop each family's declared companions to the TRIMMED NOM-HK window, merge only those streams (stray manifest files ignored), and write with family product-definition YAML (`ProductID` set from the ObsID registry). `CAL_EVENT_BY_OBSID` is derived from `libera_utils.obsids`; lunar cals remain deferred.
- Support the noise calibration (RAD ObsID 515), which `libera_utils` split out of the original combined gain/noise ObsID 512. It is a distinct calibration event producing its own `NOISE` CAL product, but combines on the existing `gain` family — same full-rate merge recipe (RAD-FULL + CAL-FULL companions on `RAD_FULL_PACKET_ICIE_TIME`, no Az/El) and the same `GAIN_product_definition.yml` template, with `ProductID` set to `NOISE` at write time.
- Track the renamed `libera_utils` calibration `DataProductIdentifier`s — `cal_lwc_temp{1,2,3}` became `cal_lwc_{310,320,335}k` (plus new `cal_lwc_{300,305}k` for ObsIDs 323/324), and `cal_lunar_cal{1,2}` became `cal_lunar_{south,north}_pole`. All five LWC blackbody temperatures now resolve to the `lwc` family.
- Consume the `libera_utils` dependency-family TRIMMED products. One TRIMMED product covers a whole family (`NOM-HK-{GAIN,SWC,LWC,SOLAR}-FAMILY-TRIMMED`) rather than one per ObsID. The exact ObsID comes from `ICIE__SW_OBSID_RAD`, not the filename. cal-combine takes one trimmed NOM-HK granule holding one ObsID and rejects a manifest carrying more than one.
- Take calibration family identity and membership from `libera_utils.obsids` rather than deriving it locally. The family key is the family TRIMMED `DataProductIdentifier`, replacing the `CalFamily` string literal and the `family_from_cal_product` prefix match, and `CAL_EVENT_BY_OBSID` is expanded from `get_family_specs`, so an ObsID added to an existing family becomes dispatchable with no libera_rad change. libera_rad still owns which streams are merged; that recipe is a subset of `get_family_inputs`, the wider set libera_cdk stages.
- SWC, LWC, and SOLAR products include SPICE-derived `Azimuth_Position` / `Elevation_Position` on `RAD_SAMPLE_FPE_TIME`, computed from AZROT-CK and ELSCAN-CK on the input manifest. cal-combine does not honor L1B's `use_geo` flag. GAIN attaches no motor attitude and needs no kernels.

## 0.6.1

- Migrate geolocation onto `curryer.compute.geometry.GeometryData`: remove the in-house SPICE subsatellite implementation (`_spacecraft_ecef_positions`, `_subsatellite_lla_from_ecef`, `calculate_libera_base_subsatellite_geolocation`) in favor of a single vectorized `calculate_geometry` call, and return instrument geolocation as one lat/lon/alt DataFrame.
- Populate `Subsatellite_Latitude`/`_Longitude`/`_Colatitude`, `Subsolar_Latitude`/`_Longitude`/`_Colatitude`, and `Radius_of_Satellite_from_Center_of_Earth`, all of which previously shipped as `-999`/`-9999` fill.
- Move `use_geo: false` fill values out of L1B packaging into `geolocation.create_placeholder_geometry`, alongside the existing azimuth/lat-lon producers; the packager now always receives a geometry DataFrame and no longer branches on its absence.
- Raise a parsed, user-facing `RuntimeError` when a curryer SPICE query fails or returns no coverage for the granule, instead of silently packaging NaN.
- Query the spacecraft and instrument observers separately in `calculate_geometry`, populating the boresight surface geometry: `Viewing_Zenith_Surface`, `Solar_Zenith_Surface`, `Viewing_Azimuth_Surface_WRT_North`, `Solar_Azimuth_Surface_WRT_North`, `Relative_Azimuth_Surface`, `Cone_Angle`, and `Cone_Angle_Rate`. Boresight fields are NaN in `jpss_only` mode, where the instrument frame does not resolve; the spacecraft fields still come through.
- Derive `Colatitude` from curryer's `surface_colatitude` rather than computing it locally.
- Validate the spacecraft and instrument observer frames against the Libera FK (`SPACECRAFT_OBSERVERS`, `INSTRUMENT_OBSERVERS`), raising `ValueError` for a frame used in the wrong role rather than silently computing geometry for the wrong optic.
- Populate the motion/attitude fields: `Satellite_Velocity`, `Satellite_Attitude_Q0..Q3`, `Clock_Angle`, `Clock_Angle_Rate`, `Along_Track_Angle`, `Cross_Track_Angle`, and `Line_Of_Sight`. Velocity and attitude ride the spacecraft observer, so they still resolve in `jpss_only`; the clock and along/cross-track angles ride the instrument observer and are NaN there.
- Add `calculate_start_of_hour_state`, populating `Satellite_Position_Start_Of_Hour` and `Satellite_Velocity_Start_Of_Hour` on the fixed 24-hour `N_HOURS` grid for the granule's start day.
- Gate `Clock_Angle_Rate` inside a 12 degree nadir cone, where the clock angle is an azimuth about nadir and its rate is a coordinate singularity rather than a real derivative. Ungated, 204 of 2995 reference samples exceeded the declared `valid_range` of `[-20, 20]`, peaking at 7588 deg/s.
- Add `libera_rad.constants` for the geometry SPICE frames and science tunables, replacing defaults previously embedded at their call sites.
- Require `lasp-curryer >= 0.5.0` for the `GeometryField` enum and the SPICE error classifier.

## 0.6.0

- Add calibration L1A combiners into `libera_rad.calibration.combiners`, remove offline `run_*_cal_event` scripts in favor of integration tests under `tests/integration/test_calibration/`, and update tests to use canonical import paths.
- Require `libera-utils >= 5.8.1` (tier-0 calibration product identifiers and definitions).
- Align calibration product identifiers with `libera_utils` naming (`GAIN-COMBINED`, `LW-TEMP*-COMBINED`, `SW-COMBINED`, `SOLAR-FACE*-COMBINED`) and update combiner time-variable selection.
- Rebuild calibration combined product definition YAMLs (gain/LW/SW/solar) and enforce strict product-definition conformance in calibration integration tests.

## 0.5.6

- Production geolocation: populate `Subsatellite_*` from spacecraft ECEF position via `spatial.ecef_to_geodetic` (attitude-independent, aligned with v0.5.5 jpss_only approach); populate motor `Azimuth`/`Elevation` from CK encoder frame; derive `Colatitude` from instrument latitude.

## 0.5.5

- Add `jpss_only` manifest configuration: load only JPSS-SPK and JPSS-CK dynamic kernels, query LIBERA*BASE spacecraft ECEF position via SPICE and derive subsatellite geolocation with `spatial.ecef_to_geodetic` (no motor CK or instrument pointing kernels), populate instrument and `Subsatellite*\*`lat/lon/colat from a single call, and write Azimuth/Elevation as 0°. Production mode derives subsatellite lat/lon from`sc_xyz_df`returned by the existing`LIBERA_SW_RAD` ellipsoid intersection. Warn when other SPICE files are listed but skipped.
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
