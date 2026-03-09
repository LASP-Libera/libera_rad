---
applyTo: "**"
---

# libera_rad Coding Instructions

## Project Overview

`libera_rad` is the LASP Libera radiometer L1B algorithm package. It ingests L1A radiometer
sample data and housekeeping files (plus SPICE kernel files) from a manifest, converts raw
digital numbers (DNs) to calibrated radiances through a seven-step pipeline, and writes a
NetCDF4 L1B output product. The primary users are LASP instrument scientists and the automated
Libera data-processing system.

- **Language/runtime:** Python >=3.11, <4
- **Dependency manager:** [Poetry](https://python-poetry.org/) — use `poetry update && poetry sync` to install/update

---

## Package Layout

| Module/Package                                 | Responsibility                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `libera_rad/l1b.py`                            | Top-level seven-step L1B algorithm orchestration; `algorithm()` entry point                           |
| `libera_rad/cli.py`                            | `argparse`-based CLI; `main()` entry point registered as `libera-rad`                                 |
| `libera_rad/geolocation.py`                    | Latitude/longitude/altitude calculation via SPICE kernels and `libera_utils`                          |
| `libera_rad/config.py`                         | Package-relative paths to bundled data files (`transfer_function.nc`, calibration JSON, product YAML) |
| `libera_rad/version.py`                        | Version string management                                                                             |
| `libera_rad/calibration/constants.py`          | `ChannelName`, `BoardName`, `DetectorType`, `RadianceMethod` enums and enum-lookup helpers            |
| `libera_rad/calibration/calibration_models.py` | Pydantic v2 models for ground-calibration data (`LiberaGroundCalibration`, etc.)                      |
| `libera_rad/radiometer/radiance.py`            | Radiance computation functions (numerical and physical methods)                                       |
| `libera_rad/radiometer/gain_calibration.py`    | FFT-based gain calibration and 100 Hz downsampling                                                    |
| `libera_rad/data/`                             | Bundled calibration JSON, product definition YAML, transfer function NetCDF                           |
| `tests/unit/`                                  | Fast, mock-based unit tests                                                                           |
| `tests/integration/`                           | End-to-end tests using real L1A/L1B data files (require LFS data)                                     |
| `calibration_tools/`                           | Offline utilities for regenerating calibration JSON; not part of the installed package                |
| `learning_notebooks/`                          | Jupyter notebooks for algorithm walkthroughs; excluded from linting                                   |

---

## Code Standards

- **Linter/formatter:** Ruff (`ruff check --fix` and `ruff format`). Line length: **120**. Enabled rule sets: `E`, `W`, `F`, `I` (isort), `S` (bandit/security), `PT` (pytest-style), `UP` (pyupgrade). Security rules (`S`) are disabled for files under `tests/`.
- **Type annotations:** Required on all function signatures. Use the `|` union syntax (PEP 604, e.g. `Path | S3Path`). Use built-in generics (PEP 585, e.g. `dict[str, xr.Dataset]`).
- **Docstrings:** NumPy style with `Parameters`, `Returns`, and `Raises` sections. All public functions and classes must have docstrings.
- **Pre-commit hooks:** Run `pre-commit install` once after cloning. Hooks enforce Ruff linting/formatting, Prettier (YAML/JSON/Markdown), trailing whitespace, large-file guard (1000 KB), AWS credential detection, private key detection, Jupyter notebook output stripping (`nbstripout`), spell-check (`codespell`), and a LASP-specific hook that **requires every `TODO[LIBSDC-123]`/`FIXME[LIBSDC-123]` comment to include a LIBSDC Jira ticket number** (e.g. `# TODO LIBSDC-123: ...`).
- **Logging:** Use `logger = logging.getLogger(__name__)` in every module; never use `print()` for operational output.
- **Security:** `bandit` runs as part of Ruff's `S` rule set. Do not disable security rules in production code.

---

## Testing

- **Framework:** `pytest` with plugins `pytest-cov`, `pytest-randomly`, `pytest-subprocess`.
- **Test locations:** Unit tests in `tests/unit/`, integration tests in `tests/integration/`, shared fixtures in `tests/conftest.py`.
- **Run unit tests only:** `pytest -m 'not integration'`
- **Run all tests (requires LFS data):** `pytest`
- **Run with coverage:** `pytest --cov=libera_rad --cov-report=html:htmlcov -m 'not integration'`
- **xfail behaviour:** `xfail_strict = true` — expected-failure tests that unexpectedly pass will fail the suite.
- **Mocking:** Use `unittest.mock` (`Mock`, `patch`). Patch at the point of use (e.g. `patch("libera_rad.l1b.smart_open")`). Use context-manager `with` blocks for multiple simultaneous patches.
- **Fixtures:** Defined in `tests/conftest.py`. Key fixtures: `calibration_data` (returns `LiberaGroundCalibration`), `test_data_path`, `test_integration_data_path`, `test_dynamic_kernels_path`, `generate_input_manifest`. Add new fixtures to `conftest.py` unless they are test-class-local.
- **Test class convention:** Group related tests in a `class Test<FunctionName>:` class with a docstring.
- **Parametrize:** Use `@pytest.mark.parametrize` for data-driven tests.
- **Integration test marker:** Mark with `@pytest.mark.integration`; CI runs these in the daily workflow.

---

## Key Patterns

- **Algorithm entry point:** Call `libera_rad.l1b.algorithm(manifest)` — it owns the full seven-step pipeline. Do not call sub-steps individually from outside the package.
- **Enums for channel/board names:** Always use `ChannelName`, `BoardName`, `DetectorType` from `libera_rad.calibration.constants` instead of raw strings. Use `get_channel_name_enum()` and `find_channel_variable()` helpers for lookups.
- **Calibration data:** Load via `LiberaGroundCalibration` Pydantic model (JSON → `**dict`). The bundled JSON path is `libera_rad.config.l1b_ground_calibration_path`. Do not hard-code file paths.
- **Bundled data paths:** Always obtain paths from `libera_rad.config` (e.g. `config.product_config_path`). Use `Path(__file__).parent / "data" / ...` pattern only inside `config.py`.
- **xarray datasets:** Read NetCDF files with `xr.open_dataset(handle).load()` (call `.load()` immediately to close the file handle). Access numpy arrays via `.values` or `.to_numpy()`.
- **S3/local paths:** Use `cloudpathlib.S3Path | pathlib.Path` throughout; use `libera_utils.smart_open` and `libera_utils.smart_copy_file` for transparent local/S3 I/O.
- **Pydantic models:** Use Pydantic v2 `BaseModel` for all structured calibration and configuration data. Do not use dataclasses or plain dicts where a schema already exists.
- **SPICE kernels:** Obtain a `KernelManager` from `libera_utils.libera_spice.kernel_manager`; never load SPICE kernels directly.

---

## Restrictions for AI Agents

**No package publishing:** Do not run `poetry publish`, `twine upload`, or any equivalent command. Publishing is handled by the LASP CI/CD pipeline.

**No git write operations:** Do not run `git commit`, `git push`, `git tag`, `git rebase`, `git merge`, or `git reset --hard`. These modify shared repository state.

**No real cloud/S3 operations:** Do not make live calls to AWS S3 or any other cloud endpoint using `boto3`, `cloudpathlib`, or the `aws` CLI. All S3 paths must be mocked in tests.

**No real SPICE kernel downloads:** Do not fetch SPICE kernels from NAIF or any remote server. Tests must use the kernel fixtures in `tests/test_data/dynamic_kernels/`.

**No credential use:** Do not read, log, or transmit any credential files, environment variables containing secrets, or AWS profiles. The pre-commit hook already blocks accidental commits of credentials.

**No modification of bundled calibration data:** Do not edit `libera_rad/data/l1b_ground_calibration.json`, `libera_rad/data/transfer_function.nc`, or `libera_rad/data/L1B_RAD-4CH_product_definition.yml` without explicit instruction. These are versioned scientific inputs.

**No Docker builds or pushes:** Do not run `docker build`, `docker push`, or `docker-compose` commands. Docker images are managed by CI.
