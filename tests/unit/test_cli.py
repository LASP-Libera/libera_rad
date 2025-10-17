"""Tests for cli module"""

# Installed
import argparse

import pytest

# Local
from libera_rad import cli, l1b


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["-v", "input_manifest.json"],
            argparse.Namespace(func=l1b.algorithm, manifest="input_manifest.json", verbose=True),
        ),
        (["--version"], argparse.Namespace(func=cli.print_version_info, manifest=None, verbose=False)),
    ],
)
def test_parse_cli_args(cli_args, parsed):
    assert dict(vars(cli.parse_cli_args(cli_args))) == dict(vars(parsed))
