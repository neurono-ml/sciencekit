#!/usr/bin/env python3
"""Validate the scikit-learn parity matrix against the DAG, the baseline and tests."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import parity_tooling  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--matrix", default=str(parity_tooling.MATRIX_PATH), help="path to parity.yaml"
)
arguments = parser.parse_args()
sys.exit(parity_tooling.main(arguments.matrix))
