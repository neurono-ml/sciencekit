# Review

Consolidated review pass over the `linear-support-vector-classifier` spec, the
`linear_svc` implementation and its companion tests. The review was executed by the
implementing agent in the same session, not by a separate reviewer; the checklist below is
the evidence a separate reviewer can re-run.

## Scenario coverage

| Requirement / scenario | Evidence |
| --- | --- |
| Builder-only construction with documented defaults | `linear_svc_builder_tests.rs::linear_svc_builder_applies_documented_defaults`, `linear_svc_config_tests.rs::linear_svc_config_carries_the_backend_and_execution_mode` |
| Backend reaches the fitted model and its prediction spans | `linear_svc_observability_tests.rs::configured_backend_is_recorded_in_fit_and_prediction_spans` |
| Separable binary data is separated | `linear_svc_fit_tests.rs::binary_fit_separates_a_separable_fixture_with_one_head` |
| Exhausted budget is reported | `linear_svc_solver_tests.rs::exhausted_iteration_budget_reports_not_converged_with_the_executed_count` |
| Loss selection and default squared hinge | `linear_svc_config_tests.rs`, `acceptance_tests/tiny_data_tests.rs::acceptance_two_rows_fit_and_predict_for_both_losses` |
| One-vs-rest three/four heads | `linear_svc_multiclass_tests.rs::three_class_fit_retains_three_heads_in_canonical_order`, `four_classes_retain_four_independent_heads` |
| Decision ties break by smallest class index | `linear_svc_predict_tests.rs::decision_ties_break_by_the_smallest_canonical_class_index` |
| Balanced and explicit class weights | `linear_svc_weights_tests.rs` |
| Decision matrix shape and argmax labels | `linear_svc_predict_tests.rs::decision_function_shape_and_labels_match_the_argmax`, `binary_decision_function_exposes_two_mirrored_columns` |
| Probabilities are unsupported | `linear_svc_predict_tests.rs::probabilistic_prediction_is_rejected_with_guidance`, `..._for_multiclass_models_too` |
| Dense/CSR parity without densification | `linear_svc_representation_tests.rs` (including the 40 × 4 000 000 fixture) |
| Determinism and concurrent prediction | `linear_svc_concurrency_tests.rs` |
| Validation taxonomy | `linear_svc_validation_tests.rs` |
| Acceptance (tiny, large sparse, metrics, weights, parallel) | `linear_svc/acceptance_tests/` |

## Findings addressed

1. **Design file exceeded the 200-line rule.** Split into `design/mod.rs` (pure dispatcher)
   plus `access.rs`, `dense.rs`, `sparse.rs` and `dispatch.rs`.
2. **Backend-span scenario had no test.** Added `linear_svc_observability_tests.rs`, which
   records every span and asserts the configured backend and execution mode on all of them.
3. **Tracing callsite-interest cache made the new span test order-dependent.** The test
   rebuilds the interest cache before installing its subscriber.
4. **Convergence tail on ill-conditioned fixtures.** Documented in the chapter and in the
   builder surface; acceptance fixtures use `tolerance(1e-2)` where 1e-4 precision is not
   needed, and the `NotConverged { iterations }` contract stays explicit.

## Remaining limitations

- LIBLINEAR shrinking is not implemented (documented as a road not taken; no builder knob).
- The intercept is recovered after the dual solve by a 1-D bisection rather than being an
  unconstrained dual coordinate, which is documented in the chapter.
- A truly independent review must be repeated by a separate agent before merge; this file
  records the coverage and the fixes it can verify.

## Independent adversarial review

Second pass, executed by a separate agent after the implementation session. Verdict matrix
below names the test that proves each scenario; `WEAK` rows were tightened in the same
review and the strengthened test is named after the arrow.

### Scenario coverage

| Requirement / scenario | Proving test | Verdict |
| --- | --- | --- |
| Builder-only construction, documented defaults | `linear_svc_builder_tests.rs::linear_svc_builder_applies_documented_defaults`; constructor is `pub(crate)` | PASS |
| Backend reaches the fitted model | `linear_svc_observability_tests.rs::configured_backend_is_recorded_in_fit_and_prediction_spans` | PASS |
| Separable binary data is separated | `linear_svc_fit_tests.rs::binary_fit_separates_a_separable_fixture_with_one_head` | PASS |
| Exhausted budget is reported | `linear_svc_solver_tests.rs::exhausted_iteration_budget_reports_not_converged_with_the_executed_count` (solver only) → `linear_svc_fit_tests.rs::fit_reports_the_exhausted_iteration_budget_through_the_public_surface` | WEAK → PASS |
| Squared hinge is the default | `linear_svc_config_tests.rs::linear_svc_config_defaults_to_squared_hinge` | PASS |
| Hinge loss is honored | `linear_svc_config_tests.rs::linear_svc_config_honors_explicit_hinge_loss` only reported the enum → `linear_svc_config_tests.rs::each_loss_minimizes_its_own_objective_better_than_the_other` | WEAK → PASS |
| Three classes produce three heads | `linear_svc_multiclass_tests.rs::three_class_fit_retains_three_heads_in_canonical_order` (also asserts each head wins its own rows) | PASS |
| Decision ties are deterministic | `linear_svc_predict_tests.rs::decision_ties_break_by_the_smallest_canonical_class_index` | PASS |
| Balanced weights change the boundary | `linear_svc_weights_tests.rs::balanced_weights_change_the_boundary_on_an_imbalanced_fixture` | PASS |
| Explicit weights are honored | `linear_svc_weights_tests.rs::explicit_class_weights_are_honored_and_change_the_fit` only asserted "changed" → `linear_svc_weight_edges_tests.rs::explicit_class_weights_reach_the_solver_in_their_own_class_role` (closed form) | WEAK → PASS |
| Decision function shape | `linear_svc_predict_tests.rs::decision_function_shape_and_labels_match_the_argmax` | PASS |
| Labels match the argmax | same test plus `binary_decision_function_exposes_two_mirrored_columns` | PASS |
| Probability request is rejected | `linear_svc_predict_tests.rs::probabilistic_prediction_is_rejected_with_guidance` and `..._for_multiclass_models_too` | PASS |
| Sparse fit equals dense fit | `linear_svc_representation_tests.rs::dense_and_csr_fits_agree_without_densifying` | PASS |
| Sparse prediction without densifying | `linear_svc_representation_tests.rs::sparse_prediction_matches_dense_prediction_and_reads_stored_entries` and `..._matrix_too_wide_to_densify` (40 × 4 000 000) | PASS |
| Repeated fits agree | `linear_svc_concurrency_tests.rs::repeated_fits_produce_identical_models` | PASS |
| Concurrent prediction agrees | `linear_svc_concurrency_tests.rs::concurrent_prediction_matches_the_sequential_result`; `acceptance_tests::parallel_equals_sequential_tests` | PASS |
| Single-class target is rejected | `linear_svc_validation_tests.rs::validation_rejects_empty_and_single_class_training_input` | PASS |
| Non-positive `C` is rejected | `linear_svc_builder_tests.rs::linear_svc_builder_rejects_invalid_hyperparameters` | PASS |
| Feature-count mismatch is rejected | `linear_svc_validation_tests.rs::validation_rejects_feature_count_mismatch_and_empty_queries` | PASS |
| Zero class weight must not break the fit | `linear_svc_weight_edges_tests.rs::explicit_zero_weight_class_is_ignored_without_breaking_convergence` | FAIL → PASS |

### Defect found and fixed

- **Zero class weight made the hinge fit diverge forever.** `SKClassWeight::Explicit`
  accepts zero weights, but the hinge dual pinned such rows at `U = 0` while the
  projected-gradient stopping rule still measured their permanently violating gradient,
  so every fit returned `NotConverged` regardless of the budget; the squared-hinge path
  also formed `1/(2·0)`. Fixed by skipping rows whose effective capacity is zero
  (`fix(svm)` commit, regression test above).

### Convention audit

| Rule | Evidence | Verdict |
| --- | --- | --- |
| `SK`/`sk_` prefixes on public items | `SKLinearSVC*`, `sk_linear_svc_loss_from_code`, module `linear_svc` | PASS |
| No abbreviations and `SK`+PascalCase | `maximum_number_of_iterations`, `regularization_c`; `SKLinearSVCLoss`/`SKLinearSVCHead` match the `SKLinearSVC*` casing of their siblings | PASS |
| Builder-only construction | `from_configuration` is `pub(crate)`; every test builds through the builder | PASS |
| `backend` default `Faer`, recorded in every public span | builder default test; observability test; `fit`, `decision_function`, `predict_labels`, `predict_probabilities` all pass `self.backend` | PASS |
| Dense/sparse parity, no densification | `design/` trait + stored-entry scans; no `to_dense` under `linear_svc/`; huge sparse fixture | PASS |
| Folder module with pure-dispatcher `mod.rs` | `linear_svc/mod.rs` and `design/mod.rs` only declare and re-export | PASS |
| `*_tests.rs` declared at the end of `mod.rs` | all companion modules registered at the end; new `linear_svc_weight_edges_tests` registered in order | PASS |
| No manual index loops | coordinate sweep uses `(0..rows).for_each` (inherently sequential); prediction uses `axis_iter_mut`; no unchecked indexing | PASS |
| Files under 200 lines | `linear_svc_weights_tests.rs` reached 207 lines during this review, so the new class-weight edge cases moved to their own companion | PASS after move |

### Numerical honesty

- The stopping rule is the maximum absolute projected gradient against `tolerance`;
  hand-computed two-point optima and the closed-form class-weighted optimum match
  exactly, and each loss attains a lower primal objective under its own loss than the
  other fit does.
- **The tolerance is absolute.** On an 8-row separable fixture with features near `100`,
  a default fit returns `NotConverged` even at `C = 0.01`, and reaching `1e-4` needs about
  `7·10^5` sweeps. This is honest (the error is explicit) and matches LIBLINEAR-style
  behavior on unscaled data. The builder's `tolerance` doc comment and the book chapter now
  state that the rule is absolute and advise standardizing features (`SKStandardScaler`)
  instead of loosening the tolerance.
- **The intercept is effectively regularized.** It enters the dual as an augmented
  constant column (LIBLINEAR-compatible) and is then re-optimized for the fixed weights; the
  bisection cannot undo the regularization that shaped those weights. On the offset fixture
  the fit lands at `w ≈ 0.00698, b ≈ -0.754` (penalized-intercept optimum), while an
  unpenalized-bias optimum is `w ≈ 0.5, b ≈ -54` (primal objective `7.45` versus `0.125`).
  The spec only requires an intercept, so this is not a scenario failure. Design Decision 4
  and the book chapter now describe the augmented, regularized intercept and its scaling
  consequence instead of claiming a purely unpenalized bias.
- Tests use hand-checkable fixtures with asserted numeric values, not only "no panic".

### Remaining non-blocking observations

- The always-true `SKLinearSVCModel::converged()` accessor was removed from the public
  surface: `fit` already returns `NotConverged` instead of building a model, so the method
  could never report anything but `true` and carried no information. Its vacuous test
  assertions were removed with it.
- The `_` match arms in `decision.rs` and `design/dispatch.rs` are required: `SKDataView` is
  `#[non_exhaustive]` in `sciencekit_common`, so the wildcard is mandatory across the crate
  boundary. Both arms carry a comment saying so.
- Model export is an explicit non-goal of this change (deferred to Wave 7 and recorded in
  `docs/PRD.md`), so PRD §8.7 acceptance is met except for export by design.
