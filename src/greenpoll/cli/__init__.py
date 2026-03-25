"""Command-line interface for GreenPoll."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build and return the GreenPoll CLI parser."""
	parser = argparse.ArgumentParser(
		prog="greenpoll",
		description="GreenPoll: flower detection and pollination planning toolkit.",
	)
	parser.add_argument(
		"--version",
		action="store_true",
		help="Print the installed GreenPoll version and exit.",
	)
	return parser


def main() -> int:
	"""Run the GreenPoll CLI entrypoint."""
	parser = build_parser()
	args = parser.parse_args()

	if args.version:
		from greenpoll import __version__

		print(__version__)
	else:
		parser.print_help()

	return 0
