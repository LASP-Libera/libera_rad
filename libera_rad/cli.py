"""Module for the Libera WFOV camera L1b processing CLI

libera-cam
"""

# Standard
import argparse

# Installed
# Local
from libera_rad import l1b
from libera_rad.version import version as libera_rad_version


def main(cli_args: list = None):
    """Main CLI entrypoint that runs the function inferred from the specified subcommand"""
    args = parse_cli_args(cli_args)
    args.func(args)


def print_version_info(*args):
    """Print CLI version information"""
    print(f"Libera radiometer science data processing CLI\n\tVersion {libera_rad_version()}")


def parse_cli_args(cli_args: list):
    """Parse CLI arguments

    Parameters
    ----------
    cli_args : list
        List of CLI arguments to parse

    Returns
    -------
    Namespace
        Parsed arguments in a Namespace object
    """
    parser = argparse.ArgumentParser(prog="libera-rad", description="Libera radiometer science data processing CLI")
    parser.add_argument(
        "--version",
        action="store_const",
        dest="func",
        const=print_version_info,
        help="print current version of the CLI",
    )

    parser.add_argument(
        "manifest",
        type=str,
        nargs="?",  # Makes manifest optional when --version is used
        help="input manifest file",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")

    # Set default function to l1b processing
    parser.set_defaults(func=l1b.algorithm)

    parsed_args = parser.parse_args(cli_args)

    # If --version wasn't used but no manifest provided, show error
    if parsed_args.func == l1b.algorithm and not parsed_args.manifest:
        parser.error("manifest file is required")

    return parsed_args
