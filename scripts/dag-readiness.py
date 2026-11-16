#!/usr/bin/env python3
"""Compute done/ready/blocked for every DAG change and the next change per lane."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dag_tooling  # noqa: E402


def read_merged(values: list[str], files: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        merged.extend(item.strip() for item in value.split(",") if item.strip())
    for file_name in files:
        text = pathlib.Path(file_name).read_text(encoding="utf-8")
        merged.extend(line.strip() for line in text.splitlines() if line.strip())
    return merged


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dag", default=str(dag_tooling.DAG_PATH), help="path to dag.yaml")
parser.add_argument(
    "--merged",
    action="append",
    default=[],
    help="comma-separated change ids already merged (repeatable)",
)
parser.add_argument(
    "--merged-file",
    action="append",
    default=[],
    help="file with one merged change id per line (repeatable)",
)
parser.add_argument("--json", action="store_true", help="print machine-readable output")
arguments = parser.parse_args()
sys.exit(
    dag_tooling.main_readiness(
        arguments.dag,
        merged=read_merged(arguments.merged, arguments.merged_file),
        as_json=arguments.json,
    )
)
