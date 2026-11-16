# Review — sparse-pairwise-distances

Reviewer: independent review pass over the `pairwise-distances` spec, the
`sciencekit_math` sparse kernels and their companion tests. The code and the openspec plan
are merged; this review closes task 8.3.

## Scenario coverage

| Requirement / scenario | Evidence |
| --- | --- |
| Sparse rows compare with sparse rows | `sparse_pairwise_dispatches_every_metric_representation_cell`, `sparse_squared_euclidean_matches_dense_on_the_same_logical_matrix` |
| Sparse rows compare with dense rows | `sparse_squared_euclidean_mixed_paths_match_dense`, `sparse_euclidean_mixed_paths_match_dense`, `sparse_manhattan_mixed_paths_match_dense`, `sparse_chebyshev_mixed_paths_match_dense`, `sparse_cosine_mixed_paths_match_dense` |
| Sparse input is never densified | `sparse_pairwise_wide_matrix_is_not_densified`, `csr_dense_does_not_densify_the_sparse_operand` |
| Squared Euclidean equals dense | `sparse_squared_euclidean_matches_dense_on_the_same_logical_matrix`, `..._accepts_f32`, `..._matches_dense_on_rectangular_inputs` |
| Empty sparse rows have zero distance | `sparse_squared_euclidean_empty_rows_have_zero_distance` |
| Symmetric with zero diagonal | `sparse_squared_euclidean_is_symmetric_with_zero_diagonal` |
| Euclidean is the square root | `sparse_euclidean_is_sqrt_of_squared` |
| Manhattan equals dense | `sparse_manhattan_matches_dense_on_the_same_logical_matrix`, `..._matches_dense_on_rectangular_inputs` |
| One-sided stored coordinates contribute absolute value | `sparse_manhattan_one_sided_coordinates_contribute_absolute_values` |
| Chebyshev equals dense | `sparse_chebyshev_matches_dense_on_the_same_logical_matrix`, `..._matches_dense_on_rectangular_inputs` |
| One-sided Chebyshev coordinate | `sparse_chebyshev_one_sided_coordinates_contribute_absolute_values`, `sparse_chebyshev_sees_one_sided_dense_maximum` |
| Identical sparse rows have zero distance | `sparse_chebyshev_identical_rows_have_zero_distance`, `sparse_manhattan_identical_rows_have_zero_distance` |
| Cosine equals dense | `sparse_cosine_matches_dense_on_the_same_logical_matrix`, `..._matches_dense_on_rectangular_inputs` |
| Zero-norm row keeps the dense policy | `sparse_cosine_zero_norm_row_keeps_the_dense_policy`, `sparse_cosine_orthogonal_rows_have_unit_distance` |
| Cosine is scale invariant | `sparse_cosine_is_scale_invariant` |
| NaN-aware Euclidean is dense-only | `sparse_pairwise_rejects_sparse_nan_euclidean`, `sparse_pairwise_nan_euclidean_is_dense_only_in_every_cell`, `sparse_pairwise_dense_nan_euclidean_keeps_the_dense_kernel` |
| Shape validation | `sparse_pairwise_rejects_feature_count_mismatch` |
| Empty inputs produce an empty matrix | `sparse_pairwise_empty_inputs_produce_empty_matrices` |
| Output is dense, inputs stay borrowed | The dispatch returns `Array2` while taking `ArrayView`/`CsMatView`; `sparse_pairwise_wide_matrix_is_not_densified` proves no densification |
| Parallel equals sequential | `sparse_pairwise_parallel_equals_sequential` (every metric via `assert_parallel_matches`) |
| Concurrent calls are deterministic | `sparse_pairwise_concurrent_calls_are_deterministic` |

## Findings

1. **Minor — NaN-aware policy.** The dense-only exception is correct per the spec
   (NaN-aware distance over implicit zeros is semantically undefined) and is covered in
   every representation cell.
2. **Minor — fixture sizes.** Large/wide fixtures exist (`sparse_pairwise_wide_matrix_is_not_densified`,
   rectangular variants); no blocker.

## Verdict

PASS — every metric is proven dense/sparse/mixed, sparse is never densified, and the
dense-only NaN-aware exception is explicit with guidance.
