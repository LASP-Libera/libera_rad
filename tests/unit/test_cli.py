"""Tests for cli module"""

# Installed
import argparse

import pytest

# Local
from libera_rad import cli, l1b
from libera_rad.calibration.combiners import cal_combine


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["-v", "input_manifest.json"],
            argparse.Namespace(func=l1b.algorithm, manifest="input_manifest.json", verbose=True),
        ),
        (["--version"], argparse.Namespace(func=cli.print_version_info, manifest=None, verbose=False, version=True)),
        (
            ["cal-combine", "cal_manifest.json"],
            argparse.Namespace(func=cal_combine.algorithm, manifest="cal_manifest.json", verbose=False),
        ),
    ],
)
def test_parse_cli_args(cli_args, parsed):
    actual = vars(cli.parse_cli_args(cli_args))
    expected = vars(parsed)
    # Only compare keys present in the expected namespace (subcommand parser omits --version).
    for key, value in expected.items():
        assert actual[key] == value
