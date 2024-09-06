"""Tests for cli module"""
# Installed
import argparse
import pytest
# Local
from libera_rad import cli, l1b, pass_through


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['l1b', '-v', 'input_manifest.json'],
         argparse.Namespace(
             func=l1b.algorithm,
             manifest='input_manifest.json',
             verbose=True
         )),
        (['pass_through', '-p', 'spice_azel', 'input_manifest.json'],
            argparse.Namespace(
                func=pass_through.algorithm,
                manifest='input_manifest.json',
                processing_step='spice_azel'
            )),
        (['pass_through', '-p', 'spice_jpss', 'input_manifest.json'],
            argparse.Namespace(
                func=pass_through.algorithm,
                manifest='input_manifest.json',
                processing_step='spice_jpss'
            )),
        (['pass_through', '-p', 'l1b_cam', 'input_manifest.json'],
            argparse.Namespace(
                func=pass_through.algorithm,
                manifest='input_manifest.json',
                processing_step='l1b_cam'
            )),
        (['pass_through', '-p', 'l1b_rad', 'input_manifest.json'],
            argparse.Namespace(
                func=pass_through.algorithm,
                manifest='input_manifest.json',
                processing_step='l1b_rad'
            )),
        (['--version'],
            argparse.Namespace(
                func=cli.print_version_info
            ))
    ]
)
def test_parse_cli_args(cli_args, parsed):
    assert dict(vars(cli.parse_cli_args(cli_args))) == dict(vars(parsed))
