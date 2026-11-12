# Independent Review — Scenario to Test Map

Every scenario in `specs/kernel-functions/spec.md` and
`specs/support-vector-machines/spec.md` is exercised by at least one named test in the
workspace. Run the gates with:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## `kernel-functions`

| Scenario | Test |
| --- | --- |
| Linear kernel is the inner product | `kernels::linear_tests::linear_kernel_computes_known_dense_dot_products_and_shape` |
| Polynomial kernel applies degree and offset | `kernels::polynomial_tests::polynomial_kernel_computes_known_values_and_matches_sparse` |
| Radial-basis-function kernel uses squared distance | `kernels::radial_basis_tests::radial_basis_function_kernel_handles_identical_and_distant_rows_and_cancellation` |
| Sigmoid kernel applies the hyperbolic tangent | `kernels::sigmoid_tests::sigmoid_kernel_computes_hyperbolic_tangent_and_matches_all_representations` |
| Precomputed kernel accepts a square matrix | `kernels::precomputed_tests::precomputed_kernel_accepts_valid_square_matrix_with_matching_rows` |
| Sparse and dense inputs agree | `kernels::linear_tests::linear_kernel_matches_across_all_four_dense_and_sparse_pairs`, `kernels::radial_basis_tests::radial_basis_function_kernel_matches_across_dense_sparse_and_mixed_views`, `kernels::sigmoid_tests::sigmoid_kernel_computes_hyperbolic_tangent_and_matches_all_representations` |
| Mixed representation is supported | `kernels::linear_tests::linear_kernel_matches_across_all_four_dense_and_sparse_pairs` |
| Sparse input is not densified | `kernels::linear_tests::linear_kernel_handles_ultra_wide_csr_without_densifying` |
| Output shape follows the inputs | `kernels::linear_tests::linear_kernel_computes_known_dense_dot_products_and_shape` |
| Inputs stay borrowed | `kernels::linear_tests::linear_kernel_inputs_stay_borrowed_and_unchanged` |
| Negative gamma is rejected | `kernels::config_tests::kernel_invalid_parameters_are_rejected_with_named_errors` |
| Non-square precomputed matrix is rejected | `kernels::precomputed_tests::precomputed_kernel_rejects_non_square_row_mismatch_and_sparse_views` |
| Feature-count mismatch is rejected | `kernels::linear_tests::linear_kernel_self_matrix_is_symmetric_and_rejects_feature_mismatch` |
| Identical rows have unit RBF similarity | `kernels::radial_basis_tests::radial_basis_function_kernel_handles_identical_and_distant_rows_and_cancellation` |
| Distant rows have negligible RBF similarity | `kernels::radial_basis_tests::radial_basis_function_kernel_handles_identical_and_distant_rows_and_cancellation` |
| Parallel equals sequential | `kernels::parallel_tests::parallel_kernel_matches_sequential_on_large_dense_and_sparse_inputs` |
| Concurrent kernel calls are deterministic | `kernels::parallel_tests::concurrent_kernel_calls_are_deterministic_across_threads` |

## `support-vector-machines`

| Scenario | Test |
| --- | --- |
| Defaults are applied without configuration | `svc::svc_builder_tests::svc_builder_applies_documented_defaults`, `svr::svr_builder_tests::svr_builder_applies_documented_defaults` |
| Backend override reaches the fitted model | `observability_and_execution_tests::configured_backend_is_recorded_in_fit_and_prediction_spans` |
| Linearly separable data is separated | `svc::svc_fit_tests::svc_fit_binary_separable_retains_support_vectors_and_separates_training_rows` |
| Support vectors are a subset of training rows | `svc::svc_fit_tests::svc_fit_binary_separable_retains_support_vectors_and_separates_training_rows` |
| Three classes produce three pairwise classifiers | `svc::svc_multiclass_tests::svc_multiclass_fits_deterministic_one_vs_one_pairs_using_only_each_pairs_rows` |
| Vote ties are deterministic | `svc::svc_predict_tests::svc_predict_breaks_hand_built_vote_ties_by_smallest_canonical_index` |
| Probabilities sum to one | `svc::svc_probability_tests::svc_probability_binary_and_multiclass_sum_to_one_and_are_monotonic` |
| Binary calibration uses Platt scaling | `probability::platt_tests::platt_five_fold_cross_validation_is_deterministic_and_monotonic`, `svc::svc_probability_tests::svc_probability_binary_and_multiclass_sum_to_one_and_are_monotonic` |
| Multiclass probabilities use pairwise coupling | `probability::coupling_tests::coupling_recovers_exact_three_class_distribution_and_differs_from_vote_counts` |
| Predictions follow the fitted response surface | `svr::svr_tests::svr_fits_smooth_one_dimensional_response_and_retains_only_support_vectors` |
| Epsilon band controls support vectors | `svr::svr_tests::svr_widening_epsilon_band_never_increases_support_vector_count` |
| Kernel choice changes the decision boundary | `acceptance_tests::tiny_and_kernels_tests::acceptance_nonlinear_xor_fixture_separates_with_rbf_better_than_linear` |
| Invalid kernel parameter is rejected | `validation_tests::validation_reports_not_converged_invalid_kernels_and_transfers_backend`, `svc::svc_builder_tests::svc_builder_honors_every_setter_and_rejects_invalid_hyperparameters` |
| Sparse fit equals dense fit | `representation_fit_tests::representation_fit_dense_and_csr_agree_for_svc_and_svr` |
| Mixed prediction is supported | `representation_predict_tests::representation_predict_supports_all_four_fit_and_query_combinations_for_svc`, `representation_predict_tests::representation_predict_supports_all_four_fit_and_query_combinations_for_svr` |
| Converged fit reports convergence | `smo::stopping_tests::smo_stopping_detects_converged_toy_problem_and_recovers_bias`, `observability_and_execution_tests::f32_svc_and_svr_fit_predict_report_iterations_and_svr_reports_not_converged` |
| Exhausted budget is reported | `smo::stopping_tests::smo_stopping_returns_not_converged_with_exact_iteration_count_on_exhaustion` |
| Repeated fits agree | `determinism_tests::determinism_repeated_fits_and_thread_counts_produce_bit_identical_models`, `determinism_tests::determinism_svr_repeated_fits_match_exactly` |
| Parallel prediction agrees with sequential prediction | `concurrency_tests::concurrency_large_batch_parallel_prediction_matches_single_row_sequential` |
| Empty input is rejected | `validation_tests::validation_rejects_empty_input_and_single_class_targets` |
| Single-class classification is rejected | `validation_tests::validation_rejects_empty_input_and_single_class_targets` |
| Non-finite value is rejected | `validation_tests::validation_rejects_shape_mismatch_and_non_finite_values` |

## Findings addressed

- Precomputed kernels now bypass the feature finiteness scan, validate the square training
  Gram matrix and the query row count up front, and reject sparse precomputed views with
  `UnsupportedRepresentation` before scanning elements.
- SVM fit and prediction resolve their `SKExecutionPlan` from the builder state, reject
  out-of-core streaming mode, and record the configured backend and interval in their
  observability spans.
- `f32` estimators run end to end and report the executed iteration count; an exhausted
  SVR budget returns `NotConverged` with that count.
- Internal identifiers use full words (`cache_size`, `diagonal`, `subset_gram`,
  `cross_validation`), and per-pair binary fitting lives in `svc/binary_fit.rs`.
- Kernel inputs stay borrowed; the dense output never aliases a source buffer.
