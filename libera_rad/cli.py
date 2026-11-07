"""Module for the Libera radiometer science data processing CLI

libera-rad
"""

# Standard
import argparse
import sys

# Local
from libera_rad import l1b
from libera_rad.calibration import cal_algorithm
from libera_rad.version import version as libera_rad_version


def main(cli_args: list = None):
    """Main CLI entrypoint that runs the function inferred from the specified subcommand"""
    args = parse_cli_args(cli_args)
    args.func(args)


def print_version_info(*args):
    """Print CLI version information"""
    print(f"Libera radiometer science data processing CLI\n\tVersion {libera_rad_version()}")


def parse_cli_args(cli_args: list | None = None):
    """Parse CLI arguments.

    Supported invocations:

    - ``libera-rad <manifest>`` — L1B (default, CDK-compatible)
    - ``libera-rad cal-combine <manifest>`` — ObsID-dispatched calibration combine
    - ``libera-rad --version``

    Parameters
    ----------
    cli_args : list, optional
        List of CLI arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    Namespace
        Parsed arguments in a Namespace object
    """
    if cli_args is None:
        cli_args = sys.argv[1:]

    # Dispatch cal-combine before the L1B positional parser so the subcommand
    # token is not consumed as a manifest path.
    if cli_args and cli_args[0] == "cal-combine":
        cal_parser = argparse.ArgumentParser(
            prog="libera-rad cal-combine",
            description="Combine L1A inputs into an ObsID-specific calibration product",
        )
        cal_parser.add_argument("manifest", type=str, help="input manifest file")
        cal_parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")
        parsed_args = cal_parser.parse_args(cli_args[1:])
        parsed_args.func = cal_algorithm.algorithm
        return parsed_args

    parser = argparse.ArgumentParser(
        prog="libera-rad",
        description="Libera radiometer science data processing CLI",
        epilog="Calibration combine: libera-rad cal-combine <manifest> (also needs LIBERA_CAL_OBSID)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print current version of the CLI",
    )
    parser.add_argument(
        "manifest",
        type=str,
        nargs="?",
        help="input manifest file",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")

    parsed_args = parser.parse_args(cli_args)

    if parsed_args.version:
        parsed_args.func = print_version_info
        return parsed_args

    parsed_args.func = l1b.algorithm
    if not parsed_args.manifest:
        parser.error("manifest file is required")
    return parsed_args
