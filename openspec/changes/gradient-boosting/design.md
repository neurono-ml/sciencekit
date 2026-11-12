# Design

## Context

See `proposal.md` — Why. `sciencekit_tree` (W4.1) grows CART structures with a response
and computes leaf values from its criterion. Boosting needs the structure grown on
pseudo-residuals but leaf values computed from gradients and Hessians, so the tree crate
must expose a structure-only build plus leaf replacement. `sciencekit_ensemble` owns the
boosting substrate shared with the histogram variant.

## Goals / Non-Goals

**Goals:**
- A loss-agnostic stage loop reusable by the histogram booster.
- Correct Newton leaf values and shrinkage.
- Binary and multiclass classification plus regression.
- Early stopping without a separate model family.

**Non-Goals:**
- Histogram binning (W4.5), categorical features, missing-value routing.
- Streaming/`partial_fit`, `memmap2` and export.
- Quantile and Poisson losses.

## Decisions

### Decision 1 — Boosting substrate in `sciencekit_ensemble`

A shared module defines `SKGradientLoss<F>`:

```
initial_prediction(targets) -> F (or a vector for multiclass)
negative_gradient(target, prediction) -> F
hessian(target, prediction) -> F
link / inverse link
loss(prediction, target) -> F
```

Classification uses the log-loss with a sigmoid (binary) or softmax (multiclass) link;
regression implements squared error, absolute error and Huber. The stage loop consumes
the trait, so the tree code and the histogram booster share it.

Alternative considered: a separate loop per loss (duplicated early stopping, shrinkage and
multiclass handling).

### Decision 2 — Structure-only tree build plus leaf replacement

The tree crate gains an advanced surface:

```
SKTreeStructure::build_structure(features, pseudo_residuals, constraints) -> structure
structure.replace_leaf_values(values: &[F]) -> Result<(), SKError>
```

The public `SKDecisionTreeRegressor` keeps its criterion-driven leaf values and does not
use this surface. Boosting builds the structure on the negative gradient with the tree's
squared-error split criterion, then replaces each leaf value with the Newton step. The
advanced surface validates the leaf count and leaves the structure unchanged on error.

### Decision 3 — Newton leaf values and shrinkage

For a leaf with gradient sum `G` and Hessian sum `H`:

```
value = -G / (H + lambda)
```

and the model adds `learning_rate * value` to the raw score. For squared error this equals
the leaf mean of the residuals; for log loss it is the Newton step. Leaves with `H = 0`
(degenerate) keep value zero.

### Decision 4 — Stage loop

For each stage:

1. compute the negative gradient and Hessian for every training row from the current raw
   score;
2. optionally subsample rows (`subsample < 1`) with the seeded generator;
3. grow a structure on the selected rows with `maximum_depth`,
   `minimum_samples_leaf` and `maximum_features`;
4. replace leaf values with the Newton step and scale by `learning_rate`;
5. add the tree to the model and update the raw scores.

Binary classification grows one tree per stage; multiclass grows one tree per class per
stage and updates each raw score with its class tree. Tree fitting is single-stage
sequential by definition; split search inside a tree is feature-parallel.

### Decision 5 — Initial prediction

Regression starts at the target mean. Binary classification starts at the log-odds of the
positive prior. Multiclass starts at the class log-priors. The initial prediction is stored
on the model and added before the stage trees.

### Decision 6 — Early stopping

When `n_iter_no_change` is set, a seeded `validation_fraction` split is held out before
fitting. After each stage the validation loss is computed; the best stage count and its
model state are retained, and fitting stops after `n_iter_no_change` non-improving stages.
The model reports the fitted stage count.

### Decision 7 — Prediction surface

Regression: `init + Σ learning_rate · tree(x)`. Binary: sigmoid of that score gives the
positive probability. Multiclass: softmax over the per-class scores. Labels use the
canonical class table and the smallest-index tie-break. Feature importances aggregate the
stage trees weighted by sample count.

### Decision 8 — Representation parity

The tree base learner accepts dense and CSR. Boosting materializes only the row subset it
fits (a row copy, not a densification) and passes it through. Prediction dispatches once
and uses the tree's dense/sparse traversal.

### Decision 9 — Validation and observability

Builder defaults and validation follow the proposal. `fit` transfers backend/execution
mode to the model; every span records the backend. Errors: `EmptyInput`,
`InvalidHyperparameter`, `ShapeMismatch`, `NonFiniteValue`, `NotConverged` (not used),
and the tree error converted into the central taxonomy.

### Decision 10 — File layout

```
crates/sciencekit_tree/src/structure/
  mod.rs, build.rs, leaf_values.rs, *_tests.rs
crates/sciencekit_ensemble/src/boosting/
  mod.rs, loss/{mod.rs, squared.rs, absolute.rs, huber.rs, log_loss.rs, *_tests.rs},
  stage.rs, early_stopping.rs, init.rs, *_tests.rs
crates/sciencekit_ensemble/src/gradient_boosting/
  mod.rs, classifier.rs, regressor.rs, builder.rs, model.rs, fit.rs, predict.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Absolute-error Hessian is degenerate** → use a constant Hessian `1` and document the
  approximation; the leaf median is not the same as the Newton step, so tests assert the
  documented behavior.
- **Multiclass uses a diagonal Hessian** → standard one-tree-per-class approximation;
  documented.
- **Early stopping needs a validation split** → seeded and reproducible; disabled when
  `n_iter_no_change` is unset.
- **Leaf replacement is an advanced API** → covered by the decision-tree delta and tests;
  public CART behavior is unchanged.
- **Stage fitting is memory-resident** → streaming is a later change.

## Migration Plan

Additive: new ensemble estimators and an advanced tree surface. No existing API changes.

## Open Questions

None blocking. Categorical features and missing routing arrive with the histogram variant.

## Deferred capabilities (task 8.3)

This change deliberately stops at the exact-split, in-memory, dense/CSR booster. Each of
the following is a separate subsystem with its own spec; none of them is silently
half-implemented — the public surface either rejects the input with a typed error or
simply does not expose the entry point yet.

| Capability | Status in this change | Why it is deferred |
|---|---|---|
| **Histogram binning** | Absent. The exact split search over sorted feature values is used instead. | Binning changes the split search, the subsample gather and the memory layout at once. Shipping exact splits first gives the histogram booster (planned as its own change, `histogram-gradient-boosting`) a validated reference implementation to compare predictions and importances against, instead of debugging bins and gradients together. The loss contract, stage loop, subsampling and early stopping are already shared, so only the structure builder changes. |
| **Categorical features** | Absent. `fit` consumes numeric dense/CSR data only. | Categorical routing needs an encoding contract (ordinal vs one-hot vs native categorical splits) that belongs to the tree crate, not to boosting. Adding it here would bake an encoding decision into the ensemble layer. |
| **Missing-value routing** | Absent, and rejected up front. Any `NaN` or infinite feature cell fails `fit`/`predict` with `SKError::NonFiniteValue` at its flattened coordinate (`gradient_boosting/validation.rs`). | Default-direction missing routing (as in scikit-learn's HistGradientBoosting) is a tree-level split policy with its own semantics and tests. Accepting `NaN` before that policy exists would silently produce arbitrary leaf assignments; the explicit rejection is the honest contract. |
| **Streaming / `partial_fit` / `fit_streaming`** | Absent. `OutOfCoreStreaming` execution intent is rejected by `sk_reject_streaming_mode` with `ExecutionModeIncompatible { mode: "out-of-core-streaming", pattern: "random-access" }`. | Gradient boosting is inherently multi-pass: every stage reads the full design again to recompute gradients and Hessians against updated raw scores, and subsampling plus early stopping need random row access. A streaming contract therefore needs a reopenable provider, a persistent raw-score store and a convergence policy — its own change. |
| **`memmap2` datasets** | Absent. `SKDataView` has only `Dense` and `Sparse` variants; there is no memory-mapped variant to dispatch on. | Memory mapping is a data-source concern shared by every estimator; it lands with the dataset/source work, not with one algorithm. Supporting it only for boosting would fragment `SKDataView`. |
| **Model export (`SKExportable` / `SKModelState`)** | Absent. The fitted models expose inspection getters (stages, trees, initial predictions, class table, importances) but do not implement `SKExportable`. | Export must serialize the additive structure, the per-stage trees and the loss metadata in a stable schema. That schema is defined by the interop/model-format work; implementing it locally would create a second, incompatible format. Until then the models remain inspectable in memory and usable in pipelines. |
| **Quantile and Poisson losses** | Absent. `SKGradientBoostingRegressionLoss` has `SquaredError`, `AbsoluteError` and `Huber`. | They are pure additions to the loss contract and can land later without touching the stage loop; they were left out to keep this change's validation matrix focused. |

## Self-review: scenario to test matrix (task 8.4)

Every scenario in `specs/gradient-boosting/spec.md` and `specs/decision-trees/spec.md` was
mapped to at least one named executable test. The review found one real gap in the
acceptance suite — multiclass predictions were compared against raw integer labels while
`SKClassifierPredictor::predict_labels` is documented to return canonical class indices —
which was fixed in `gb_large_tests.rs` by canonicalizing the truth labels with
`sk_canonicalize_labels` before scoring.

| Spec scenario | Named test (all passed) |
|---|---|
| Default configuration | `classifier_builder_defaults_match_contract`, `regressor_builder_defaults_match_contract` |
| Backend reaches the fitted model | `separable_binary_fixture_and_backend_span_propagation`, `concurrent_prediction_agrees_with_sequential_and_records_regressor_backend` |
| Squared-error gradient | `squared_error_loss_matches_hand_computed_contract` |
| Log-loss gradient | `binary_log_loss_matches_hand_computed_log_odds_gradients_and_hessians` |
| Multiclass gradients | `multiclass_log_loss_matches_hand_computed_softmax_and_class_vectors`, `multiclass_single_stage_fits_one_tree_per_class` |
| One-leaf Newton step | `newton_leaf_value_matches_hand_computed_step_and_degeneracy_guard`, `hand_computed_binary_fit_and_zero_score_sigmoid` |
| Learning rate scales the update | `single_stage_stump_and_split_apply_newton_leaves_and_shrinkage`, `hand_computed_single_stage_updates_for_all_three_regression_losses` |
| Regression sums the stages | `hand_computed_single_stage_updates_for_all_three_regression_losses`, `smooth_response_recovered_under_every_regression_loss` |
| Binary probabilities are sigmoid of the score | `hand_computed_binary_fit_and_zero_score_sigmoid` |
| Multiclass probabilities sum to one | `multiclass_fits_one_tree_per_class_per_stage_and_softmax_sums_to_one` |
| Subsample is reproducible | `stochastic_row_and_feature_subsampling_are_reproducible_and_diversify`, `classifier_and_regressor_dense_and_sparse_fit_and_predict_are_equivalent` |
| Full subsample uses every row | `full_subsample_uses_every_training_row` |
| Early stop reports fewer stages | `scalar_binary_and_multiclass_loops_stop_early_and_restore_best_stages`, `large_regression_gradient_boosting_all_losses_early_stop_and_metrics` |
| Disabled early stopping fits every stage | `scalar_binary_and_multiclass_loops_stop_early_and_restore_best_stages` (`full.stages.len() == 12`) |
| Importances aggregate the stages | `two_stages_with_different_importances_produce_sample_weighted_aggregate`, `classifier_importances_normalize_and_zero_split_boosters_return_zeros` |
| Sparse fit equals dense fit | `classifier_and_regressor_dense_and_sparse_fit_and_predict_are_equivalent` |
| Sparse prediction is supported | `classifier_and_regressor_dense_and_sparse_fit_and_predict_are_equivalent`, `ultra_wide_sparse_never_densifies_and_csc_is_rejected` |
| Non-positive learning rate is rejected | `invalid_classifier_hyperparameters_are_rejected`, `invalid_regressor_hyperparameters_are_rejected`, `hyperparameter_bounds_and_early_stop_validation_are_enforced` |
| Invalid subsample is rejected | `hyperparameter_bounds_and_early_stop_validation_are_enforced` |
| Single-class classification is rejected | `single_class_non_finite_values_and_streaming_modes_are_rejected` |
| Repeated fits agree | `repeated_fits_and_rayon_thread_counts_produce_identical_models` |
| Concurrent prediction agrees | `concurrent_prediction_agrees_with_sequential_and_records_regressor_backend` |
| Boosting replaces leaf values | `build_structure_and_replace_leaf_values_preserve_split_geometry` |
| Leaf replacement validates length | `replace_leaf_values_validates_length_and_leaves_structure_unchanged` |
| Public estimator behavior is unchanged | `public_decision_tree_regressor_is_unaffected_by_structure_surface` |

Additional coverage beyond the written scenarios: `empty_input_and_shape_mismatches_are_rejected`
(dimensions), the tiny and 20 000-row acceptance suites (PRD §8.7: little data, lots of data,
concurrency, dense/sparse and metrics through `sciencekit_metrics`), and the `f32`/`f64` paths in
`configured_values_reach_classifier_in_f64_and_f32` and
`configured_values_reach_regressor_in_f64_and_f32`.

**Verdict:** every scenario is exercised by a passing named test; the only uncovered surface is the
deferred capability list above, which is contractual rather than missing.
