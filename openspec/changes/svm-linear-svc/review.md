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
