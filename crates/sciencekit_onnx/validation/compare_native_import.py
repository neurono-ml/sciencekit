#!/usr/bin/env python3
"""Regenerate the external ONNX fixtures and check them with onnxruntime.

Each fixture is a plain ONNX (default-domain) graph produced by the Python ONNX
tooling, standing in for a model exported by another runtime:

    <name>.onnx            the graph
    <name>.input.json      {"input": [[float, ...], ...]}
    <name>.expected.json   onnxruntime output(s) plus the comparison tolerance

Run:
    python3 validation/compare_native_import.py <fixture-directory>

The Rust companion test `native_external_fixture_tests` imports the same files
and compares the native prediction against `expected.json`, so this script is the
reference-runtime half of the parity check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def check(directory: Path, name: str) -> list[str]:
    model_path = directory / f"{name}.onnx"
    input_path = directory / f"{name}.input.json"
    expected_path = directory / f"{name}.expected.json"
    onnx.checker.check_model(onnx.load(str(model_path)))
    probe = np.asarray(json.loads(input_path.read_text())["input"], dtype=np.float32)
    expected = json.loads(expected_path.read_text())
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"X": probe})))
    failures = []
    tolerance = float(expected.get("tolerance", 1e-6))
    for key in ("prediction", "probabilities", "labels"):
        if key not in expected:
            continue
        reference = np.asarray(expected[key], dtype=np.float64)
        actual = np.asarray(outputs["label" if key == "labels" else key], dtype=np.float64)
        if reference.shape != actual.shape:
            failures.append(f"{name}: {key} shape {actual.shape} != {reference.shape}")
            continue
        difference = np.max(np.abs(reference - actual)) if reference.size else 0.0
        if difference > tolerance:
            failures.append(f"{name}: {key} max |difference| {difference:.3e} > {tolerance:.1e}")
    return failures


def main(root: Path) -> int:
    failures = []
    for model_path in sorted(root.glob("*.onnx")):
        failures.extend(check(root, model_path.stem))
    for failure in failures:
        print(failure)
    print(f"{len(list(root.glob('*.onnx')))} fixtures checked, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "validation/external")))
