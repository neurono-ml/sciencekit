#!/usr/bin/env python3
"""Render or check the documented parity coverage page from parity.yaml."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import parity_tooling  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--matrix", default=str(parity_tooling.MATRIX_PATH))
parser.add_argument("--output", default=str(parity_tooling.COVERAGE_PAGE))
parser.add_argument("--check", action="store_true", help="fail when the page is stale")
arguments = parser.parse_args()
sys.exit(
    parity_tooling.main_render(arguments.matrix, arguments.output, check=arguments.check)
)
