#!/usr/bin/env python3
"""Validate the implementation DAG: ids, dependencies, lanes, waves and completion."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dag_tooling  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dag", default=str(dag_tooling.DAG_PATH), help="path to dag.yaml")
parser.add_argument("--design", default=str(dag_tooling.DESIGN_PATH), help="path to design.md")
parser.add_argument("--no-design", action="store_true", help="skip the design sync check")
arguments = parser.parse_args()
sys.exit(
    dag_tooling.main_check(
        arguments.dag, None if arguments.no_design else pathlib.Path(arguments.design)
    )
)
