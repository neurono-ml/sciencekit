# Independent Review — Native Model Import

Task 4.4 verification of the `native-model-import` change.

- Code branch: `feat/native-model-import` (PR #394)
- OpenSpec branch: `feat/native-model-import-openspec` (PR #244)
- Documentation branch: `docs/native-model-import-book` (PR #395)
- Reviewer: separate agent, did not write the implementation.
- Reference runtime: `onnx 1.23.2` + `onnxruntime 1.32.0`.
- Gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`, `mdbook build docs`,
  `openspec validate native-model-import --strict` — all green (tails in §6).

## 1. Reference-runtime harness

Command:

```text
python3 crates/sciencekit_onnx/validation/compare_native_import.py \
  crates/sciencekit_onnx/validation/external
```

Real output:

```text
2 fixtures checked, 0 failures
```

Both external fixtures (`external_regressor`, `external_classifier`) pass
`onnx.checker.check_model` and re-execute in `onnxruntime` on the same probe.

## 2. Hostile input matrix

Every probe was executed from a separate binary linked against the reviewed
worktree; panics were caught with `catch_unwind`. No probe panicked and no probe
returned a half-built model.

| Attack | Result |
|---|---|
| Safetensors header length `u64::MAX` | `Err(TruncatedFile { format: "safetensors", needed: 18446744073709551615, available: 8 })` |
| Header length `u64::MAX - 4` (8-byte prefix overflows) | `Err(TruncatedFile { needed: 18446744073709551615, available: 10 })` |
| Header length `u64::MAX / 2` | `Err(TruncatedFile …)` |
| JSON state declaring `SKStandardScaler` but carrying linear tensors | `Err(MissingTensor { name: "standard_scaler.mean" })` |
| JSON state of a family outside the subset (`SKLasso`) | `Err(UnsupportedFamily { estimator_name: "SKLasso" })` |
| Valid Safetensors of a foreign model (Hugging Face header) | `Err(UnsupportedFamily …)` — no `sciencekit.format_version` |
| Valid but empty foreign Safetensors | `Err(UnsupportedFamily …)` |
| ONNX input dtype `F64` with `F32` weights | `Err(DtypeMismatch { tensor: "X", F32, F64 })` |
| ONNX integer (`I64`) affine weights | `Err(DtypeMismatch { tensor: "W", F64, I64 })` |
| ONNX node reading an undeclared initializer | `Err(InvalidGraph { "node 0 reads ghost …" })` |
| ONNX missing MatMul weight initializer | `Err(InvalidGraph { "the model declares no usable prediction head …" })` |
| ONNX declared input width 2 with 3-column weights | `Err(ShapeMismatch { tensor: "X", [3], [2] })` |
| ONNX zero-column weight `[0, 1]` | `Err(ShapeMismatch { tensor: "feature_count", [1], [0] })` |
| ONNX zero-row affine graph | `Err(InvalidGraph { "the MatMul weight payload disagrees with its shape" })` |
| ONNX tree branch mode `BRANCH_GT` / `BRANCH_LT` / `BRANCH_GTE` / `BRANCH_EQ` | `Err(UnsupportedOperator { operator: "TreeEnsembleRegressor", … })` |
| ONNX tree `missing_value_tracks_true = 1` | `Err(UnsupportedOperator …)` |
| JSON state with `feature_count = 0` | `Err(ShapeMismatch { tensor: "feature_count", [1], [0] })` |

## 3. Numerical equivalence — three seeds

A standalone probe re-fits each family on seeded data (`seeds 11, 29, 47`),
exports to Safetensors and to JSON, imports both, and compares predictions with
`==` on the raw values (no tolerance) against the fitted model.

Families covered: `SKLinearRegression`, `SKRidge`, `SKSGDRegressor`,
`SKLogisticRegression` (labels and probabilities), `SKSGDClassifier`,
`SKStandardScaler`, `SKMinMaxScaler`, `SKRobustScaler`,
`SKDecisionTreeRegressor`, `SKDecisionTreeClassifier` (labels and
probabilities), `SKRandomForestRegressor`, `SKRandomForestClassifier` (labels
and probabilities), `SKKMeans`.

```text
ALL EXACT: true
```

No exactness claim failed, for Safetensors or for JSON, on any of the three
seeds. ONNX export/import was also probed on the same seeds: linear, logistic,
the three scalers, decision tree regressor and classifier are bit-exact; the
random-forest regressor differs by at most `4.77e-7` in `f32`, inside the
tolerance the companion test documents (the ONNX operator sums trees while the
native forest averages them).

## 4. Tree leaf threshold (`NaN` → `0`)

The exporter now writes zero at every leaf threshold (the original NaN could
not serialize). A probe fits a tree, asserts that all leaf thresholds are
exactly `0`, rebuilds the same state with `NaN` at those positions, imports both
states and the exported Safetensors file, and compares full prediction vectors:

```text
nodes=3 leaves=2 zero_leaf_thresholds=2
expected=[1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
nan_state=[1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
zero_state=[1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
lossless=true
file_lossless=true
```

Traversal (`tree_structure.rs`) reads a threshold only when `split_feature` is
`Some`; the importer reconstructs a split only for `split_feature >= 0`. A leaf
is therefore never routed by its threshold, so the encoding is lossless by
construction as well as by measurement. The native tree rejects non-finite
input, so no leaf is reachable through a missing-value path; the probe also
covers a tree whose leaves sit on both sides of the `feature <= 0.0` boundary.

## 5. Convention audit

| Rule | Verdict |
|---|---|
| `SK` / `sk_` prefixes on public items | PASS — new state/graph errors are `SKOnnxNativeError` variants and new free helpers carry `sk_`. |
| No abbreviations | PASS — the reviewer renamed the new `sk_dtype_mismatch` helper to `sk_element_type_mismatch`. |
| Builder-only construction | PASS — `SKOnnxNativeImporter::builder()`, states rebuilt through `SKModelState::new`. |
| `backend()` recorded in spans | PASS — `native_onnx_records_the_requested_backend`. |
| Dense/sparse parity | PASS after fix — the facade now takes `SKDataView`; `native_validation_prediction_accepts_dense_and_sparse_features` proves a sparse probe equals the dense one. |
| Folder modules with a pure-dispatcher `mod.rs` | PASS — `prediction.rs`, `state/scalers/mod.rs`, `state/trees/mod.rs`, `sciencekit_onnx/src/native/mod.rs` declare and re-export only. |
| `*_tests.rs` companions at the end | PASS — test modules moved to the end of both dispatchers. |
| No manual index loops in new code | PASS — the logistic weight transpose uses `flat_map`; tree rebuilds keep their pre-existing indexed reads over parallel ONNX arrays. |
| No `unwrap` on untrusted bytes | PASS — container parsing, header decoding and ONNX mapping propagate typed errors; `unwrap` appears only in test fixtures. |

## 6. Defects fixed during review

| Commit | Defect |
|---|---|
| `fix(interop): saturate the declared safetensors header length` | `8 + header_length` overflowed on an attacker-chosen `u64` and panicked in debug builds. |
| `test(interop): prove the zero leaf threshold is lossless` | Documents the zero leaf threshold and pins the invariant with a companion test. |
| `fix(interop): reject ONNX tree semantics the native routing cannot express` | `BRANCH_GT`/`BRANCH_LT`/`BRANCH_GTE`/`BRANCH_EQ` and `missing_value_tracks_true` were silently flattened into `feature <= threshold`, changing predictions. |
| `fix(interop): import ONNX forest regressors natively` | Member trees were rebuilt without `feature_importances`, so forest import always failed. |
| `fix(interop): reject zero-width models and mismatched ONNX parameters` | Zero-feature states produced unusable models; integer initializers and input-width disagreements were promoted silently instead of raising `DtypeMismatch`/`ShapeMismatch`. |
| `fix(interop): accept dense and sparse views in the prediction facade` | The facade took `ArrayView2` only, so imported models lost the fitted model's sparse parity. |
| `refactor(interop): keep native modules on the project conventions` | Test modules moved to the end; classifier surface documented; manual transpose loop replaced by iterators. |
| `refactor(interop): spell out the element type in the validation helper` | Removed the new `dtype` abbreviation. |

Every fix started from a failing companion test (the failure was observed before
the implementation landed), and no equality assertion was weakened.

## 7. Gate tails

```text
cargo fmt --all --check                  -> exit 0 (no diff)
cargo clippy --workspace --all-targets -- -D warnings
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.87s
  -> exit 0
cargo test --workspace
  test result: ok. …; TEST_EXIT=0
mdbook build docs
  INFO HTML book written to …/docs/book
  -> exit 0
openspec validate native-model-import --strict
  Change 'native-model-import' is valid
  -> exit 0
```

## Verdict

The change meets the `native-model-import` specification. All hostile inputs
return typed errors without panicking or leaving a partial model, all three
seeds import with exact predictions where exactness is claimed, and the leaf
threshold encoding is provably lossless. Approved for merge pending CI.
