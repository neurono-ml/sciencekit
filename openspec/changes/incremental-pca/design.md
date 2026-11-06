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
