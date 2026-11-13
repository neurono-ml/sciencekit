#!/usr/bin/env python3
"""Validate exported sciencekit ONNX graphs with the reference runtime.

Every fixture directory written by the Rust test suite contains:

    model.onnx     the graph produced by `SKToOnnx::to_onnx`
    input.json     {"input_name": str, "input": [[float, ...], ...]}
    expected.json  {"outputs": {name: value}, "tolerance": float}

This script loads each model with `onnx`, checks it with `onnx.checker`, runs it
with `onnxruntime` on the same input and compares every declared output with the
Rust prediction within the recorded tolerance.

Usage:
    python3 validation/compare_with_onnxruntime.py <fixture-directory> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def compare_outputs(expected: dict, actual: list, tolerance: float) -> list[str]:
    failures = []
    if set(expected) != set(actual):
        return [f"output names differ: expected {sorted(expected)}, got {sorted(actual)}"]
    for name, reference in expected.items():
        reference_array = np.asarray(reference, dtype=np.float64)
        actual_array = np.asarray(actual[name], dtype=np.float64)
        if reference_array.shape != actual_array.shape:
            failures.append(
                f"{name}: shape mismatch {reference_array.shape} != {actual_array.shape}"
            )
            continue
        difference = np.max(np.abs(reference_array - actual_array)) if reference_array.size else 0.0
        if difference > tolerance:
            failures.append(f"{name}: max |difference| {difference:.3e} > {tolerance:.1e}")
    return failures


def validate_fixture(directory: Path) -> bool:
    model = onnx.load(str(directory / "model.onnx"))
    onnx.checker.check_model(model)
    input_payload = json.loads((directory / "input.json").read_text())
    expected_payload = json.loads((directory / "expected.json").read_text())

    session = ort.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
    input_array = np.asarray(input_payload["input"], dtype=np.float32 if _is_float_model(model) else np.float64)
    feeds = {input_payload["input_name"]: input_array}
    output_names = [output.name for output in session.get_outputs()]

    if "runtime_error_contains" in expected_payload:
        needle = expected_payload["runtime_error_contains"]
        try:
            session.run(output_names, feeds)
        except Exception as error:  # noqa: BLE001 - the runtime error type is not stable
            if needle not in str(error):
                print(f"FAIL {directory.name}: the runtime failed with an unexpected error: {error}")
                return False
            print(f"PASS {directory.name}: the runtime rejected the input as expected")
            return True
        print(f"FAIL {directory.name}: the runtime accepted an input it should reject")
        return False

    values = session.run(output_names, feeds)
    actual = dict(zip(output_names, [value.tolist() for value in values]))

    failures = compare_outputs(
        expected_payload["outputs"], actual, float(expected_payload.get("tolerance", 1e-5))
    )
    if failures:
        for failure in failures:
            print(f"FAIL {directory.name}: {failure}")
        return False
    print(f"PASS {directory.name}: opset {model.opset_import[0].version}, {len(model.graph.node)} nodes")
    return True


def _is_float_model(model: onnx.ModelProto) -> bool:
    element_type = model.graph.input[0].type.tensor_type.elem_type
    return element_type == onnx.TensorProto.FLOAT


def main(arguments: list[str]) -> int:
    directories = [Path(argument) for argument in arguments]
    fixtures = [
        path if path.is_dir() else path.parent
        for root in directories
        for path in ([root] if (root / "model.onnx").exists() else sorted(root.iterdir()))
    ]
    if not fixtures:
        print("no fixture directories found")
        return 1
    results = [validate_fixture(directory) for directory in fixtures]
    print(f"{sum(results)}/{len(results)} fixtures matched the reference runtime")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
