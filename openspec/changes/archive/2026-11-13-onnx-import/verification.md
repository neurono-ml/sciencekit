# Independent verification

Reviewer: separate agent, task 4.5. Code worktree `feat/onnx-import` (PR #386),
OpenSpec worktree `feat/onnx-import-openspec` (PR #246), docs worktree
`docs/onnx-import-book` (PR #387).

## Verdict matrix

| Requirement / review item | Evidence | Verdict |
|---|---|---|
| Reference-runtime harness | `validation/compare_with_onnxruntime.py` over 36 generated fixtures | **36/36 runtime, 36/36 import** |
| Unsupported operator | `an_unsupported_operator_is_rejected_with_its_name` | pass |
| Wrong opset | `an_opset_above_the_supported_ceiling_is_named` | pass |
| Missing initializer | `a_node_reading_an_undefined_value_names_the_missing_tensor` (now rejected at load) | pass |
| Inconsistent shapes / initializer width | `an_initializer_width_that_differs_from_the_declared_input_width_is_rejected` | pass |
| Dtype mismatch | `an_integer_feature_input_is_rejected_as_an_unsupported_dtype`, `a_float16_feature_input_is_rejected_as_an_unsupported_dtype`, `f64_predictor_explicitly_converts_to_an_f32_model_input` | pass |
| External-data model | `an_external_initializer_is_rejected_by_name` | pass |
| Cyclic graph | `a_graph_with_a_cycle_is_rejected_with_the_cycle_path` | pass |
| Malformed operands/axes (no panic) | `operand_tests.rs` (2), `shape_validation_tests.rs` (5) | pass |
| Round trip, linear/pipeline/scaler/tree | `round_trip_tests.rs` (4) | pass |
| Round trip, SVM RBF + multiclass | `round_trip_families_tests.rs` | pass |
| Round trip, PCA / TruncatedSVD | `round_trip_families_tests.rs` | pass |
| Round trip, scaler `with_mean=false` | `round_trip_families_tests.rs` | pass |
| Round trip, dense KNN | `round_trip_families_tests.rs` | pass |
| Round trip, histogram gradient boosting | exporter refuses the family (`UnsupportedOperation`) | **not exportable** |
| Round trip, sparse-trained KNN | exporter refuses CSR training design (`UnsupportedOperation`) | **not exportable** |
| Backend recorded in span | `the_configured_backend_reaches_the_prediction_span`, `the_predictor_reports_the_default_backend_and_execution_mode` | pass |
| `serde-onnx` pin is `rev`, not branch | `Cargo.toml`: `rev = "0c54592"`, lock resolves `0c545921…` | pass |
| Sentinel shim narrow + tested | `decode_tests.rs` (4) | pass |

## Rejection cases added

Seven constructs are now rejected with a typed `SKOnnxImportError` and no panic,
each with a companion test: a graph cycle (`GraphCycle`, with the node path),
duplicated node outputs (`InvalidModel`), an initializer width that disagrees
with the declared input width (`ShapeMismatch`), malformed `Transpose`
permutations and `Squeeze` axes (`InvalidOperator`), a `Reshape` whose dimensions
overflow (`InvalidOperator`), and a node with a missing or empty operand
(`InvalidOperator`). The structural validator from the shared IR
(`serde_onnx::ir::validate_model`) runs at load time, so a cycle is no longer
misreported as a missing initializer.

## Round-trip evidence

Native Rust model → `to_onnx()` → `SKOnnxPredictor` → prediction, compared with
the original model at the tolerances the export tests already document
(1e-9 for f64 families, labels compared exactly): binary RBF SVC and multiclass
RBF SVC (labels and one-vs-one decision values), PCA, TruncatedSVD, standard
scaler with `with_mean=false`, and a dense-trained KNN classifier (labels and
probabilities).

Two families the review named cannot round-trip **because export refuses them**,
and the tests assert the typed refusal instead of pretending: histogram gradient
boosting (`sk_onnx_coverage` marks the boosting family unsupported; the
dispatcher returns `UnsupportedOperation`) and a sparse-trained KNN (the
neighbour exporter translates only a dense training design and names CSR in the
refusal). This is an `onnx-export` coverage decision, not an import defect.

## The exact upstream `serde-onnx` bug

Pinned revision `0c54592157183cad53c250d89eaddaf80b86a895` (v0.1.1), file
`src/proto/codec.rs`, `decode_attribute`: presence of a scalar attribute is
inferred from the encoded protobuf value (`if proto.f != 0.0`, `if proto.i != 0`,
`if !proto.s.is_empty()`) instead of from `AttributeProto.type`. Protobuf omits
scalar fields equal to their default, so a legal `INT 0` (`axis = 0`,
`keepdims = 0`), `FLOAT 0.0` or empty `STRING` looks absent; when no other field
is set the decoder returns `ProtoCodecError::Protobuf("attribute \"<name>\"
carries no value")`. Fix: select the value from `AttributeProto.type` and accept
the default value for that type. `decode_tests.rs` proves the pinned decoder
rejects the same bytes the shim decodes, and that the shim only restores the
default without touching non-zero attributes.

## Defects fixed

1. Cyclic graphs now fail at load with `GraphCycle`; structural validation also
   rejects duplicated outputs and undefined values before inference.
2. Missing/empty node operands are validated instead of indexing `node.inputs`
   (a panic on untrusted bytes).
3. `MatMul` inner dimensions, `Transpose` permutations, `Squeeze` axes and
   `Reshape` dimension overflow are validated before calling ndarray routines
   that panic.
4. Unsupported interface element types (e.g. `FLOAT16`) produce
   `UnsupportedDtype` instead of a generic central error.
5. `SKOnnxPredictorBuilder` gained `backend`/`execution_mode`; every public
   prediction records the configured kind in its `sciencekit.operation` span.
6. The in-test runtime harness now writes `imported.json`, so
   `onnx_runtime_validates_an_exported_scaler` exercises all three comparisons
   instead of failing on a missing file.

## Gates

- `cargo fmt --all --check`: pass.
- `cargo clippy --workspace --all-targets -- -D warnings`: pass.
- `cargo test --workspace`: pass.
- `mdbook build docs` (docs worktree): pass.
- `openspec validate onnx-import --strict`: `Change 'onnx-import' is valid`.
