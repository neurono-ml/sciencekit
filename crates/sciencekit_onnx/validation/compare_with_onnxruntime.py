#!/usr/bin/env python3
"""Validate exported sciencekit ONNX graphs with the reference runtime.

Every fixture directory written by the Rust test suite contains:

    model.onnx     the graph produced by `SKToOnnx::to_onnx`
    input.json     {"input_name": str, "input": [[float, ...], ...]}
    expected.json  {"outputs": {name: value}, "tolerance": float}
    imported.json  (optional) {"outputs": {name: {"shape": [...], "data": [...]}}}

The file is written by `cargo run -p sciencekit_onnx --example validate_import
<fixture-root>`. This script loads each model with `onnx`, checks it with
`onnx.checker`, runs it with `onnxruntime` on the same input, and compares:

1. the Rust prediction recorded in `expected.json` against `onnxruntime`;
2. the imported sciencekit prediction in `imported.json` against `expected.json`;
3. the imported sciencekit prediction against `onnxruntime`.

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


def compare_outputs(expected: dict, actual: dict, tolerance: float, label: str) -> list[str]:
    failures = []
    if set(expected) != set(actual):
        return [f"{label}: output names differ: expected {sorted(expected)}, got {sorted(actual)}"]
    for name, reference in expected.items():
        reference_array = np.asarray(reference, dtype=np.float64)
        actual_array = np.asarray(actual[name], dtype=np.float64)
        if reference_array.shape != actual_array.shape:
            failures.append(
                f"{label}: {name}: shape mismatch {reference_array.shape} != {actual_array.shape}"
            )
            continue
        difference = np.max(np.abs(reference_array - actual_array)) if reference_array.size else 0.0
        if difference > tolerance:
            failures.append(f"{label}: {name}: max |difference| {difference:.3e} > {tolerance:.1e}")
    return failures


def read_imported(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    outputs = {}
    for name, tensor in payload["outputs"].items():
        outputs[name] = np.asarray(tensor["data"], dtype=np.float64).reshape(tensor["shape"])
    return outputs


def validate_fixture(directory: Path) -> tuple[bool, bool]:
    """Return (runtime matched, import matched)."""
    model = onnx.load(str(directory / "model.onnx"))
    onnx.checker.check_model(model)
    input_payload = json.loads((directory / "input.json").read_text())
    expected_payload = json.loads((directory / "expected.json").read_text())

    session = ort.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
    input_array = np.asarray(
        input_payload["input"], dtype=np.float32 if _is_float_model(model) else np.float64
    )
    feeds = {input_payload["input_name"]: input_array}
    output_names = [output.name for output in session.get_outputs()]

    if "runtime_error_contains" in expected_payload:
        needle = expected_payload["runtime_error_contains"]
        try:
            session.run(output_names, feeds)
        except Exception as error:  # noqa: BLE001 - the runtime error type is not stable
            if needle not in str(error):
                print(f"FAIL {directory.name}: the runtime failed with an unexpected error: {error}")
                return False, False
            imported = read_imported(directory / "imported.json")
            message = "rejected as expected" if imported is None else "import also rejected as expected"
            print(f"PASS {directory.name}: the runtime {message}")
            return True, imported is None
        print(f"FAIL {directory.name}: the runtime accepted an input it should reject")
        return False, False

    values = session.run(output_names, feeds)
    actual = dict(zip(output_names, [value.tolist() for value in values]))
    tolerance = float(expected_payload.get("tolerance", 1e-5))
    failures = compare_outputs(expected_payload["outputs"], actual, tolerance, "runtime")

    imported = read_imported(directory / "imported.json")
    import_failures: list[str] = []
    if imported is None:
        import_failures.append("import: no imported.json was produced")
    else:
        import_failures.extend(compare_outputs(expected_payload["outputs"], imported, tolerance, "import"))
        import_failures.extend(compare_outputs(actual, imported, tolerance, "import-runtime"))

    for failure in failures + import_failures:
        print(f"FAIL {directory.name}: {failure}")
    if not failures and not import_failures:
        print(
            f"PASS {directory.name}: opset {model.opset_import[0].version}, "
            f"{len(model.graph.node)} nodes, runtime + import matched"
        )
    return not failures, not import_failures


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
    runtime_matched = sum(1 for runtime, _ in results if runtime)
    import_matched = sum(1 for _, imported in results if imported)
    print(
        f"{runtime_matched}/{len(results)} fixtures matched the reference runtime; "
        f"{import_matched}/{len(results)} fixtures matched the imported prediction"
    )
    return 0 if runtime_matched == len(results) and import_matched == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
