# Independent Review — ONNX Export

Task 4.5 verification of the `onnx-export` change.

- Code branch: `feat/onnx-export` (PR #378)
- OpenSpec branch: `feat/onnx-export-openspec` (PR #245)
- Documentation branch: `docs/onnx-export-book` (PR #379)
- Reviewer: separate agent, did not write the implementation.
- Gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`, `mdbook build docs`,
  `openspec validate onnx-export --strict` — all green.
- Reference runtime: `onnx 1.23.2` + `onnxruntime 1.32.0`, checked with
  `crates/sciencekit_onnx/validation/compare_with_onnxruntime.py`.

## 1. Reference-runtime validation

Every fixture directory writes `model.onnx`, `input.json` and `expected.json` from the
Rust test that owns the estimator; the harness runs `onnx.checker.check_model` and
re-executes the graph in `onnxruntime` on the same input.

**Result: 36/36 fixtures matched.** 35 predictive fixtures plus one negative fixture
(a wider query matrix must be rejected by the runtime). Ten fixtures were added by this
review: non-default-gamma SVC, a multiclass SVC vote tie, an explicit-class-weight
`SKLinearSVC`, an f32 SVC, Lasso, ElasticNet, `SKSGDRegressor`, `SKMiniBatchKMeans`,
a neighbour distance tie, and the feature-count mismatch.

## 2. Coverage-matrix honesty

Every row marked `supported` now has at least one passing runtime fixture, and each
estimator name in those rows has its own fixture. Rows marked unsupported refuse with a
reason naming the family.

| Coverage row | Estimators | Runtime fixtures | Verdict |
|---|---|---|---|
| `preprocessing.scalers` | Standard, MinMax, Robust | `standard_scaler_f64`, `minmax_scaler_f32_clipped`, `robust_scaler_f64` | PASS |
| `preprocessing.encoders` | OneHot | `one_hot_encoder_f64_integer` | PASS |
| `neighbors` | KNN classifier/regressor | `knn_classifier_f64`, `knn_regressor_f64`, `knn_classifier_distance_tie_f64` | PASS |
| `clustering` | KMeans, MiniBatchKMeans | `kmeans_f64`, `minibatch_kmeans_f64` | PASS (after fix) |
| `decomposition` | PCA, TruncatedSVD | `pca_f64`, `truncated_svd_f64` | PASS |
| `svm` | SVC, SVR, LinearSVC | `svc_binary_rbf_f64`, `svc_binary_linear_f64`, `svc_binary_rbf_gamma_f64`, `svc_multiclass_rbf_f64`, `svc_multiclass_vote_tie_f64`, `svc_binary_rbf_f32`, `svr_rbf_f64`, `linear_svc_binary_f64`, `linear_svc_multiclass_f64`, `linear_svc_binary_weighted_f64` | PASS |
| `composites` | Pipeline classifier/regressor | `pipeline_classifier_f64`, `pipeline_regressor_f64` | PASS |
| `trees` | Decision tree/forest classifier/regressor | four `*_f32` fixtures | PASS |
| `linear.regressors` | LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor | `linear_regression_f64`, `ridge_single_alpha_f32`, `lasso_single_alpha_f64`, `elastic_net_single_alpha_f64`, `sgd_regressor_f64` | PASS |
| `linear.classifiers` | LogisticRegression | `logistic_regression_binary_f64`, `logistic_regression_multiclass_f64` | PASS |
| `outlier`, `boosting` | none | none | PASS — explicit refusal naming the family |

`SKLinearSVC` with **negative** class weights cannot reach the exporter: the estimator
builder rejects a negative `class_weight` entry (existing `InvalidHyperparameter`), so the
attack was exercised with explicit positive weights instead, whose fitted heads the graph
reproduces exactly.

## 3. Operator semantics against the ONNX specification

| Check | Finding | Evidence |
|---|---|---|
| SVM one-vs-one vote and confidence | The graph reproduces `sk_aggregate_one_vs_one_confidences` term by term: `Greater(margin, 0)` votes the positive class, sums the signed margin, and emits `votes + margin_sum / (3·(|margin_sum|+1))`. | `svm/classification.rs`, `svm_tests` multiclass fixtures |
| `ArgMax` polarity for binary SVC | `Concat([-m, m])` then `ArgMax(axis=1)` selects index 1 iff `m > 0`; at `m == 0` both Rust (`decision > 0`) and ONNX (first index) choose class 0. | `svc_binary_rbf_f64`, `svc_binary_linear_f64` |
| Multiclass `ArgMax` tie | ONNX `ArgMax` defaults to the first maximum (`select_last_index=0`), matching the Rust winner rule "smallest class index". | `svc_multiclass_vote_tie_f64` (votes 1-1-1 → class 0) |
| `Softmax` axis | `axis=1`, one row per sample, matching the Rust per-row softmax. | `logistic_regression_multiclass_f64` |
| `TreeEnsemble` thresholds and missing routing | `BRANCH_LEQ` with the fitted threshold is the left branch, exactly the Rust `x <= t`. `nodes_missing_value_tracks_true` is all-zero, so a NaN takes the false (right) branch, matching Rust where `NaN <= t` is false. | `tree/layout.rs`, `tree/ensemble.rs`, tree/forest fixtures |
| `TreeEnsemble` attribute precision | ONNX-ML stores threshold/leaf attributes as 32-bit floats; the exporter rejects f64 trees instead of silently truncating. | `f64_trees_are_rejected_because_the_onnx_ml_attributes_are_floats` |
| `Cast` and dtype | `Cast(to=element_type::<F>())` uses the model scalar (FLOAT/DOUBLE), never a hardcoded type; initializers and the declared input/output agree with `SKFloat`. | f32 SVC fixture declares FLOAT and matches; `ridge_single_alpha_f32`, scaler and tree f32 fixtures |
| Initializer consistency | `add_floating_initializer` derives the tensor dtype from `F` and the builder validates byte length against the declared shape. | `graph_tests` byte-length and raw-f32 tests |

## 4. Defects found and fixed

| # | Defect | Fix commit |
|---|---|---|
| 1 | `robustness_tests.rs` existed but was never declared, so the metadata, dtype, size and concurrency acceptance tests silently never ran. | `fix(interop): register the robustness acceptance tests` |
| 2 | A corrupted state whose metadata feature count disagreed with its learned tensors produced a graph that only failed at prediction time; no exporter validated the declared width. | `fix(interop): reject a state whose learned width disagrees with the declared input` |
| 3 | `SKMiniBatchKMeans` was advertised as supported but every export failed: the exporter read `kmeans.cluster_centers` while the mini-batch state stores `minibatch_kmeans.centers`. | `fix(interop): export MiniBatchKMeans from its own centers tensor` |
| 4 | `svm/mod.rs` and `tree/mod.rs` carried implementations, contrary to the pure-dispatcher folder-module convention; the move also added SVM/tree width and tree-array validation and made the precomputed-kernel refusal run before tensor reads. | `fix(interop): keep the exporter folder modules as pure dispatchers` |

The harness gained a `runtime_error_contains` mode so negative fixtures assert that the
declared input width is enforced by the runtime instead of only by Rust.

## 5. Spec scenario matrix

| Requirement / scenario | Evidence | Verdict |
|---|---|---|
| Export returns bytes | every `to_onnx` test plus `onnx.checker` | PASS |
| Unsupported model is explicit | coverage tests name the family and point at `sk_onnx_coverage` | PASS |
| Inputs and outputs are typed | `assert_regressor_interface` / `assert_classifier_interface`, dtype tests | PASS |
| Learned tensors become initializers | `assert_eq!(onnx.initializer_count(), 2)` and per-family graphs | PASS |
| f32 and f64 mapping | `dtype_mapping_...` plus the f32 SVC/tree/scaler fixtures | PASS |
| Linear, tree, pipeline operators | MatMul/Add/link, ONNX-ML ensembles, composed stages | PASS |
| Classifier metadata | `classifier_metadata_carries_the_class_table` (now executed) | PASS |
| Exported graph loads and predicts | 36/36 runtime fixtures | PASS |
| Missing tensor / invalid shape is rejected | `export_rejects_a_state_whose_feature_count...`, `*_is_rejected_by_name`, runtime negative fixture | PASS |

Residual observations (not defects against this change's spec): the ONNX output label is the
canonical class index, not the class label text (documented behavior of the Rust
predictors); the tree ensemble emits an unused `tree_label` value beside the `ArgMax`
label; the harness skips inside `cargo test` when the host Python lacks `onnxruntime` and
is authoritative only when run explicitly.

**Verdict: the change meets the `onnx-export` spec.**
