# Design

## Context

See `proposal.md` — Why. `sciencekit_decomposition` is new. `sciencekit_math` supplies the
full SVD/eigh backend and the randomized SVD kernel (this wave). `sciencekit_common`
supplies `SKUnsupervisedFit`, `SKFeatureTransformer` (for transform), builder state, data
views and observability. Centering is the key constraint: it turns a sparse design dense.

## Goals / Non-Goals

**Goals:**
- A correct PCA with a solver portfolio and a clear `Auto` policy.
- Deterministic components and signs.
- Sparse support for the randomized solver through implicit centering.
- Explicit, never silent, densification for the dense-only solvers.

**Non-Goals:**
- Sparse full/eigh solvers (they need a dense centered design or covariance).
- Kernel PCA, incremental PCA (W5.6), GPU, `memmap2`, export.
- Whitening beyond the standard component scaling.
- Missing values.

## Decisions

### Decision 1 — New `sciencekit_decomposition` crate

The crate owns PCA, TruncatedSVD and IncrementalPCA. PCA stores the mean, components and
variance state. It implements `SKUnsupervisedFit` and `SKFeatureTransformer` (output
`Array2<F>`); `transform`/`inverse_transform` are inherent methods on the model.

### Decision 2 — Centering and representation policy

`Xc = X - mean` is the mathematical object every solver needs. For dense input it is a
dense matrix. For sparse input it would be dense, so:

- `Randomized` passes the mean to `sk_randomized_svd`, which computes the centered
  products implicitly and never materializes `Xc`;
- `Full` and `CovarianceEigh` reject sparse input with `UnsupportedRepresentation` unless
  the builder's `allow_densification(true)` is set;
- when `allow_densification` is true, the estimator densifies explicitly, emits a warning
  with the estimated dense size and proceeds.

There is no silent densification. `SKTruncatedSVD` remains the uncentered sparse
alternative, and the decomposition family covers sparse data without a dense copy.

### Decision 3 — Full SVD solver

Use the backend `svd` on the centered design. Components are the first `k` rows of `Vᵀ`;
explained variance is `s² / (n − 1)`; the ratio divides by the total variance. The full
solver is deterministic; any sign ambiguity is fixed by the same convention as the
randomized kernel (largest-magnitude component positive).

### Decision 4 — Randomized solver

Use `sk_randomized_svd` with `number_of_power_iterations`, `random_seed` and the feature
mean as the optional centering vector. Dense input is centered explicitly (or implicitly
by the kernel); sparse input uses the kernel's implicit centered products. The result is
already truncated; components and singular values are stored directly.

### Decision 5 — Covariance eigensolver

Form `C = Xcᵀ Xc / (n − 1)`, run the backend `eigh`, sort eigenvalues descending and take
the first `k` eigenvectors as components. This is efficient when `n_samples` is much larger
than `n_features`.

### Decision 6 — Auto policy

```
if number_of_components >= 0.8 * min(n_samples, n_features) → Full
else if n_samples > 10 * n_features and n_features <= 1000 → CovarianceEigh
else → Randomized
```

The policy is deterministic and exposed on the model (`resolved_solver`). It is a
performance heuristic, not a statistical claim.

### Decision 7 — Transform, inverse transform and whitening

`transform(X) = (X − mean) · componentsᵀ`; with whitening, divide each component by
`sqrt(explained_variance)` (or the singular-value spread). `inverse_transform(Z)` multiplies
by `components` and adds the mean. The model stores the mean and the scaling vector.

### Decision 8 — Validation and observability

Builder validation: components in `1..=min(n, d)`, tolerance ≥ 0, power iterations > 0,
valid solver. Fit validates non-empty input and finiteness; transform validates the feature
count. A sparse `Full`/`CovarianceEigh` fit requires `allow_densification(true)`; the
warning reports the estimated dense bytes. Spans record the backend; fit transfers it to
the model.

### Decision 9 — File layout

```
crates/sciencekit_decomposition/src/
  lib.rs
  pca/{mod.rs, builder.rs, config.rs, centering.rs, solvers.rs, core_implementation.rs, model.rs, fit.rs, transform.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Centering memory** → the randomized solver avoids the dense copy; full/eigh require
  explicit densification or a future sparse eigensolver.
- **Auto heuristic** → exposed and tested; an explicit solver always wins.
- **Randomized approximation** → power iterations and a tolerance in tests; full SVD remains
  available.
- **Sign ambiguity** → fixed convention shared with the randomized kernel.
- **Whitening numerical stability** → explained variance is floored before division and
  tests cover near-zero variance.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. Sparse PCA and kernel PCA are future changes.
