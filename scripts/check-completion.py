#!/usr/bin/env python3
"""Run one change's completion checks: OpenSpec status, change tests and global gates."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dag_tooling  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--change", required=True, help="change id from dag.yaml")
parser.add_argument("--dag", default=str(dag_tooling.DAG_PATH), help="path to dag.yaml")
parser.add_argument(
    "--skip-global",
    action="store_true",
    help="skip the workspace-wide gates recorded in policy.merge_requires",
)
parser.add_argument(
    "--dry-run", action="store_true", help="print the commands without running them"
)
arguments = parser.parse_args()
sys.exit(
    dag_tooling.main_completion(
        arguments.change,
        dag_path=arguments.dag,
        include_global=not arguments.skip_global,
        dry_run=arguments.dry_run,
    )
)
