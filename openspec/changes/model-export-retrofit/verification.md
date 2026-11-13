# Independent Review — Model Export Retrofit

Task 7.4 verification of the `model-export-retrofit` change.

- Code branch: `feat/model-export-retrofit` (PR #372)
- OpenSpec branch: `feat/model-export-retrofit-openspec` (PR #243)
- Documentation branch: `docs/model-export-retrofit-book` (PR #373)
- Reviewer: separate agent, did not write the implementation.
- Gates (`CARGO_TARGET_DIR=/home/claudino/.cache/sciencekit-target-model-export`,
  `CARGO_PROFILE_DEV_DEBUG=0`): `cargo fmt --all --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace`, `mdbook build docs`,
  `openspec validate model-export-retrofit --strict`.

## 1. Family coverage matrix

`cargo test` names are the executable proof. "Round-trip" means fit → export → import →
assert equality of the model's real output (`predict`, `transform`, `predict_labels`,
`predict_probabilities`, `score_samples`), not merely an `Ok` import.

| Family / model type | Export | Round-trip test | Verdict |
|---|---|---|---|
| `SKStandardScalerModel` | yes | `standard_scaler_round_trips_predictions_and_backend` | PASS |
| `SKMinMaxScalerModel` | yes | `minmax_scaler_round_trips_predictions_and_range` | PASS |
| `SKRobustScalerModel` | yes | `robust_scaler_round_trips_predictions_and_configuration` | PASS |
| `SKOneHotEncoderModel` | yes | `one_hot_encoder_round_trips_indicator_output`, `..._integer_category_codes` | PASS |
| `SKPolynomialFeaturesModel` | yes | `polynomial_features_round_trips_expansion` | PASS |
| `SKSimpleImputerModel` | yes | `simple_imputer_round_trips_strategy_and_statistics`, `..._composed_indicator` | PASS |
| `SKMissingIndicatorModel` | yes | `missing_indicator_round_trips_scope_and_scope_policy` | PASS |
| `SKKNNImputerModel` | yes | `knn_imputer_round_trips_state` | PASS |
| `SKLinearRegressionModel` | yes | `linear_regression_round_trips_coefficients_rank_and_backend` | PASS |
| `SKRidgeModel` | yes | `ridge_round_trips_multi_alpha_solutions` | PASS |
| `SKLassoModel` | yes | `lasso_round_trips_multi_alpha_solutions` | PASS |
| `SKElasticNetModel` | yes | `elastic_net_round_trips_the_mixing_ratio` | PASS |
| `SKLogisticRegressionModel` | yes | `logistic_regression_round_trips_binary_predictions`, `..._probabilities` | PASS |
| `SKSGDClassifierModel` | yes | `sgd_classifier_round_trips_classes_and_heads` | PASS |
| `SKSGDRegressorModel` | yes | `sgd_regressor_round_trips_schedule_and_iterate` | PASS |
| `SKKNeighborsClassifierModel` | yes | `knn_classifier_round_trips_dense_training` | PASS |
| `SKKNeighborsRegressorModel` | yes | `knn_regressor_keeps_a_sparse_design_sparse` (`training_representation() == "csr"`) | PASS |
| `SKSVCModel` | yes | `svc_round_trips_pairwise_support_vectors_and_classes` | PASS |
| `SKSVRModel` | yes | `svr_round_trips_support_vectors_and_kernel` | PASS |
| `SKLinearSVCModel` | yes | `linear_svc_round_trips_heads_and_class_table` | PASS |
| `SKDecisionTreeClassifierModel` | yes | `decision_tree_classifier_round_trips_nodes_and_predictions` | PASS |
| `SKDecisionTreeRegressorModel` | yes | `decision_tree_regressor_round_trips_nodes_and_predictions` | PASS |
| `SKRandomForestClassifierModel` | yes | `random_forest_classifier_round_trips_members_and_oob` | PASS |
| `SKRandomForestRegressorModel` | yes | `random_forest_regressor_round_trips_trees` | PASS |
| `SKGradientBoostingClassifierModel` | yes | `gradient_boosting_classifier_round_trips_classes_and_stages` | PASS |
| `SKGradientBoostingRegressorModel` | yes | `gradient_boosting_regressor_round_trips_stages` | PASS |
| `SKHistGradientBoostingClassifierModel` | yes | `histogram_classifier_round_trips_categorical_masks` (categorical table) | PASS |
| `SKHistGradientBoostingRegressorModel` | yes | `histogram_regressor_round_trips_binning_and_missing_routing` (bins + NaN routing), `..._a_sparse_design` | PASS |
| `SKKMeansModel` | yes | `kmeans_round_trips_centers_labels_and_backend` | PASS |
| `SKMiniBatchKMeansModel` | yes | `minibatch_model_export_and_import_round_trip_preserves_state_and_predictions` | PASS |
| `SKDBSCANModel` | yes | `dbscan_round_trips_labels_and_nearest_core_assignment` | PASS |
| `SKPCAModel` | yes | `pca_round_trips_components_mean_and_whitening` | PASS |
| `SKIncrementalPCAModel` | yes | `acceptance_incremental_pca_streaming_epochs_export_concurrency_and_metrics` | PASS |
| `SKTruncatedSVDModel` | yes | `truncated_svd_round_trips_components_and_projection` | PASS |
| `SKIsolationForestModel` | yes | `isolation_forest_round_trips_trees_offset_and_scores` | PASS |
| `SKBaggingClassifierModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKBaggingRegressorModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKGridSearchCVClassifierModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKGridSearchCVRegressorModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKHalvingGridSearchClassifierModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKHalvingGridSearchRegressorModel` | yes | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKPipelineClassifier` / `SKPipelineRegressor` | yes | none — import returns typed `UnsupportedOperation` (documented) | **FAIL (round-trip)** |
| `SKFeatureUnionModel` | yes (static tuple) | none — import returns typed `UnsupportedOperation` | **FAIL (round-trip)** |
| `SKPipelineGraphModel` | no — typed `UnsupportedOperation` | none | **FAIL (export)** |
| `SKFittedStreamingPipeline` | no — typed `UnsupportedOperation` | none | **FAIL (export)** |

Totals: **35 families** prove fit → export → import → identical output; **10 composite
model types** are explicit-unsupported; **2 type-erased pipeline types** cannot export at
all. The claim in the task ("EVERY estimator family survives the round trip") is therefore
**not met**; the composite gaps are review findings, not nits.

### Secondary scenario checks

| Check | Proving test | Verdict |
|---|---|---|
| Correct feature count is validated | KMeans/PCA/TruncatedSVD/ensemble import shape checks; `pca_import_rejects_a_mean_with_the_wrong_feature_count` | PASS |
| Sparse training stays sparse | `knn_regressor_keeps_a_sparse_design_sparse`, `histogram_regressor_round_trips_a_sparse_design` | PASS |
| NaN/missing routing | `simple_imputer_round_trips_strategy_and_statistics`, `histogram_regressor_round_trips_binning_and_missing_routing` | PASS |
| Categorical masks / binning tables | `histogram_classifier_round_trips_categorical_masks`, `histogram_regressor_round_trips_binning_and_missing_routing` | PASS |
| Imported model reports exported backend in its span | `imported_isolation_forest_reports_the_exported_backend_in_its_span` (new) | PASS |
| Documented tolerance | zero (exact equality); recorded in `design.md` Decision 5a | PASS |

## 2. Format robustness

| Corruption | Behaviour | Verdict |
|---|---|---|
| Truncated Safetensors header/payload | typed `Conversion` (existing `safetensors_validation_rejects_truncated_files`) | PASS |
| Future container version | typed `UnsupportedRepresentation` (existing shard/single-file tests) | PASS |
| Future model format version | typed `UnsupportedRepresentation`; new `linear_regression_import_rejects_a_future_format_version` | PASS |
| Required tensor missing | typed `InvalidHyperparameter`; `linear_regression_rejects_a_missing_tensor`, `ridge_import_rejects_a_coefficient_shape_mismatch`, lasso/one-hot guards | PASS |
| Unexpected extra tensor | ignored forward-compatibly, model still imports; new `linear_regression_import_ignores_an_unexpected_extra_tensor` | PASS |
| Truncated stage-importance tensor | previously silently truncated; now typed `ShapeMismatch` (defect 1) | PASS |
| Wrong-length PCA / truncated-SVD secondary tensor | previously accepted and could panic on `transform`; now typed `ShapeMismatch` (defect 2) | PASS |

No panic path was found after these fixes.

## 3. Defects found and fixed (TDD: failing test first)

1. **Truncated stage-importance tensors were silently accepted.** `gradient_boosting` and
   `histogram_boosting` importers used `values.get(offset..end).unwrap_or(&[])`, so a short
   tensor produced empty importance rows instead of a typed error. Fixed in `99c28b6`; guards
   `gradient_boosting_import_rejects_a_truncated_stage_importance_tensor`,
   `histogram_import_rejects_a_truncated_stage_importance_tensor`. The same commit validates
   the top-level `feature_importances` length.
2. **Decomposition importers accepted wrong-length secondary tensors.** PCA and truncated SVD
   stored `mean`/`singular_values`/`explained_variance*`/`whiten_scale` without checking their
   lengths against the component matrix, which panics later while broadcasting during
   `transform`. Fixed in `ae7cf8b`; guards
   `pca_import_rejects_a_mean_with_the_wrong_feature_count`,
   `truncated_svd_import_rejects_singular_values_with_the_wrong_length`.
3. **Composite model imports were a compile-time gap, not the spec's typed refusal.**
   Bagging, grid-search and successive-halving models implemented `SKExportable` but not
   `SKImportable`, so a consumer could not express the documented limitation. Fixed in
   `785ba69` with `UnsupportedOperation { operation: "import_state", reason: <guidance> }`;
   guards in the bagging/grid/halving companions.
4. **Feature-union export was missing for the static tuple form.** Fixed in `9d50a1c`
   (`feature_union/export.rs`, flat tuple exporter + `union_export_tests`).
5. **Graph and streaming pipelines had no export surface at all.** Fixed in `daca5d8` with
   explicit typed `UnsupportedOperation` for their type-erased stages, plus the feature-union
   import refusal; guards `graph_pipeline_export_returns_a_typed_unsupported_operation`,
   `streaming_pipeline_export_returns_a_typed_unsupported_operation`,
   `feature_union_import_returns_a_typed_unsupported_operation`.
6. **No test proved the imported model's observability span carries the exported backend.**
   Added in `23a91d7` for the isolation forest.
7. **Missing adversarial import coverage.** Added in `1c46a4d` (future version, extra tensor).

## 4. Convention audit (`AGENTS.md`)

| Rule | Verdict | Evidence |
|---|---|---|
| `SK` / `sk_` public prefixes | PASS | Every new public struct/trait is `SK*`; free readers are `sk_*`; methods are unprefixed. |
| No abbreviations in public names | WEAK (accepted) | `sk_feature_union_member_count` is spelled out; inherited `KNN`, `SVC`, `SVR`, `PCA`, `SVD`, `DBSCAN` names predate this change and match the plan. |
| Builder-only construction | PASS | No public constructors were added; importers reconstruct through validated fields. |
| Folder modules with pure-dispatcher `mod.rs` | PASS | New/edited `mod.rs` files only declare, re-export and register companions. |
| Companion `*_tests.rs` declared last | PASS | New tests are companions; inline test blocks stay at file ends. |
| No manual index loops | WEAK (accepted) | Two silent-truncation loops were removed, but histogram tree import still indexes node arrays by position (a record layout, not a collection traversal). |
| Files ≤ 200 lines | WEAK (accepted) | `ensemble/random_forest/export.rs` (411), `gradient_boosting/export.rs` (329), `histogram_boosting/export.rs` (315), `logistic_regression/model/export.rs` (304) exceed the threshold; splitting them is pure restructuring and was not done during review. |

## 5. Gates

```
cargo fmt --all --check                                  -> clean
cargo clippy --workspace --all-targets -- -D warnings    -> clean
cargo test --workspace                                   -> see run log (all suites green after re-run; one
                                                            pre-existing tracing-interest flake observed once in
                                                            isolation_fit_builds_seeded_ensemble_and_transfers_model_state)
mdbook build docs                                         -> HTML book written
openspec validate model-export-retrofit --strict          -> Change 'model-export-retrofit' is valid
```

## 6. Residual findings (blocking the "every family round-trips" claim)

1. Bagging, grid-search and successive-halving models still do not round-trip. Their states
   would be reconstructible only if each nested member/winner metadata were embedded; today
   they return a typed refusal. This is the honest behaviour for their current state, but it
   means the change's headline claim is false.
2. The typed pipeline, feature union, graph pipeline and streaming pipeline are export-only
   or export-refusing. Their stage chains contain configured estimator values or type-erased
   boxes, so import needs an architectural change (a serialisable stage descriptor), not a
   test.
3. The 200-line folder-module rule and the histogram node-array index loops remain style debt.
