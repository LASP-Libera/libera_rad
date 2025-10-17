# Version Changes

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
