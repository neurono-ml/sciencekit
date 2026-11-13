# Design

## Context

See `proposal.md` — Why. `sciencekit_decomposition` (W5.4/W5.5) owns the decomposition
crate. `sciencekit_common` supplies `SKLazySource`, `SKSourceProvider`, the prefetch
executor and `SKFeatureBlock`. The accumulator must accept dense and sparse batches without
retaining the data.

## Goals / Non-Goals

**Goals:**
- Exact incremental covariance merging and a finalize step equivalent to full PCA.
- Dense and sparse batch support without densification.
- Deterministic multi-epoch streaming.

**Non-Goals:**
- Full SVD (PCA owns it), missing values, GPU, `memmap2`, export.
- A rank-adaptive component count.

## Decisions

### Decision 1 — Chan-Golub-LeVeque state

The state is `(n, mean, scatter)` where `scatter = Σ (x − mean)(x − mean)ᵀ`. Merging a batch
with `(n_b, mean_b, scatter_b)` uses:

```
n = n_a + n_b
delta = mean_b - mean_a
mean = mean_a + delta * n_b / n
scatter = scatter_a + scatter_b + delta deltaᵀ * n_a n_b / n
```

This is exact (up to floating point) and never retains the old rows.

### Decision 2 — Dense and sparse accumulation

Dense batches use the merge directly. Sparse batches accumulate the Gram matrix `XᵀX` and
the column sums, then derive the scatter as `Gram − n · mean · meanᵀ`; this avoids
densifying the batch. Both paths produce the same covariance within the documented
tolerance. The scatter/Gram is `p × p`, which is the expected decomposition memory.

### Decision 3 — Finalize and components

`covariance = scatter / (n − 1)`, run the backend `eigh`, sort eigenvalues descending and
take the first `number_of_components` eigenvectors as components. Explained variance is
the eigenvalue; the ratio divides by the total variance. The sign convention matches PCA.

### Decision 4 — Batch and streaming fitting

`fit` chunks into `batch_size` blocks and calls the same update. `fit_streaming` drives
`SKSourceProvider` through the shared executor; one-shot providers run one pass, reopenable
providers run `maximum_number_of_epochs` passes. `partial_fit` requires
`batch_size >= number_of_components` (the standard restriction) so the covariance update is
numerically meaningful.

### Decision 5 — Transform and inverse transform

`transform(X) = (X − mean) · componentsᵀ`, with whitening scaling; inverse multiplies by
`components` and adds the mean. Both accept dense and sparse views; sparse products never
densify.

### Decision 6 — Validation and observability

Builder validation: components > 0, batch size > 0, valid whiten. Fit validates empty
batches, component count ≤ min(n, d), batch size ≥ components, finite values and feature
count consistency. Spans record the backend; fit transfers it to the model.

### Decision 7 — File layout

```
crates/sciencekit_decomposition/src/incremental_pca/
  mod.rs, builder.rs, config.rs, accumulator.rs, merge.rs, core_implementation.rs,
  model.rs, fit.rs, streaming.rs, transform.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Gram-based sparse accumulation loses some numerical stability** → compensated sums and
  a dense-vs-sparse tolerance test; dense uses the direct merge.
- **`batch_size >= n_components` restriction** → documented and validated.
- **Memory is `O(p²)`** → inherent to covariance PCA; sparse data is not densified.
- **Multi-epoch streaming requires a reopenable provider** → the one-shot case is explicit.
- **No adaptive component count** → future change.

## Migration Plan

Additive: new estimator in `sciencekit_decomposition`. No existing API changes.

## Open Questions

None blocking. A rank-adaptive accumulator is a future change.

## Review — scenario to test mapping

Independent review (task 4.5) mapped every scenario in
`specs/incremental-pca/spec.md` to a named test in the final code. No gaps were found;
the only correction was the `number_of_components` default wording above (the code
resolves `min(n_samples, n_features)`, not the bare feature count).

| Spec scenario | Named test |
| --- | --- |
| Default configuration | `incremental_builder_carries_documented_defaults` |
| Two batches equal one batch | `incremental_merge_two_batches_equal_one_batch_and_persist_state`, `incremental_merge_hand_checked_two_batch_mean_and_scatter` |
| State persists across calls | `incremental_merge_two_batches_equal_one_batch_and_persist_state` |
| Finalized components are sorted | `incremental_finalize_sorted_components_and_equals_full_pca` |
| Incremental equals full PCA | `incremental_finalize_sorted_components_and_equals_full_pca`, `incremental_fit_chunks_dense_and_sparse_and_matches_pca` |
| One-pass streaming | `incremental_streaming_one_shot_provider_consumes_one_pass_and_matches_fit` |
| Multi-epoch streaming | `incremental_streaming_reopenable_multi_epoch_reopens_and_preserves_subspace` |
| Batch-size invariance | `incremental_concurrency_batch_size_invariance_holds_for_dense_and_sparse` |
| Transform shape | `incremental_transform_shape_roundtrip_and_sparse_dense_parity` |
| Whitening normalizes variance | `incremental_transform_whitening_normalizes_unit_variance_and_inverts` |
| Sparse equals dense | `incremental_sparse_matches_dense_across_batches_without_densification`, `incremental_sparse_mixed_with_dense_batches_matches_single_dense_pass` |
| Small batch is rejected | `incremental_validation_rejects_excessive_components_and_small_batches` |
| Feature-count mismatch is rejected | `incremental_validation_rejects_feature_mismatch_non_finite_and_csc` |
| Repeated fits agree | `incremental_concurrency_repeated_fits_and_thread_counts_are_bit_identical` |
| Concurrent transform agrees | `incremental_concurrency_shared_model_transforms_match_sequential` |
| Acceptance (tiny/large/wide/metrics) | `acceptance_incremental_pca_tiny_f32_large_dense_and_wide_sparse`, `acceptance_incremental_pca_streaming_epochs_export_concurrency_and_metrics` |

