#!/usr/bin/env python3
"""Render the generated layer and lane blocks in the implementation DAG design."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dag_tooling  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dag", default=str(dag_tooling.DAG_PATH))
parser.add_argument("--design", default=str(dag_tooling.DESIGN_PATH))
parser.add_argument("--check", action="store_true", help="fail when the design is stale")
arguments = parser.parse_args()

dag = dag_tooling.load_dag(arguments.dag)
design = pathlib.Path(arguments.design)
if arguments.check:
    errors = dag_tooling.design_sync_errors(dag, design)
    for error in errors:
        print(error)
    sys.exit(1 if errors else 0)

text = design.read_text(encoding="utf-8")
text = dag_tooling.replace_block(text, dag_tooling.LAYERS_MARKER, dag_tooling.render_layers(dag))
text = dag_tooling.replace_block(text, dag_tooling.LANES_MARKER, dag_tooling.render_lanes(dag))
design.write_text(text, encoding="utf-8")
print(f"rendered {design}")
