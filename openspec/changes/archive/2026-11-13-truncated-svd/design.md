# Design

## Context

See `proposal.md` — Why. `sciencekit_math` supplies the randomized SVD kernel from this
wave. `sciencekit_decomposition` supplies the crate from PCA (W5.4). TruncatedSVD is the
uncentered counterpart of PCA and the sparse path of the decomposition family.

## Goals / Non-Goals

**Goals:**
- A sparse-friendly low-rank factorization without centering.
- A transform/inverse-transform surface consistent with PCA.
- Deterministic signs and seeded projections.

**Non-Goals:**
- Centering, ARPACK/IRLBA, missing values, GPU, `memmap2`, export.
- A full-SVD solver (that is PCA's `Full`).

## Decisions

### Decision 1 — Reuse the randomized SVD kernel

Fit calls `sk_randomized_svd` on the input as-is. Components are the first `k` rows of
`Vᵀ`; singular values come from the kernel. The model stores no mean. This is the direct
difference from PCA and the reason sparse input is supported.

### Decision 2 — Transform and inverse transform

`transform(X) = X · componentsᵀ`; `inverse_transform(Z) = Z · components`. Both accept
dense or sparse views and dispatch once. The sparse path uses stored entries; no dense copy
of the input is created.

### Decision 3 — Explained variance

`explained_variance` is the variance of each transformed component over the training rows;
`explained_variance_ratio` divides it by the total variance of the input (the sum of the
input column variances). The model stores both.

### Decision 4 — Validation and observability

Builder validation: components in `1..=min(n, d)`, iterations > 0, tolerance ≥ 0, algorithm
`Randomized`. Fit validates non-empty input and finiteness; transform validates the feature
count. Spans record the backend; fit transfers it to the model.

### Decision 5 — File layout

```
crates/sciencekit_decomposition/src/truncated_svd/
  mod.rs, builder.rs, config.rs, core_implementation.rs, model.rs,
  fit.rs, transform.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Randomized approximation** → power iterations and tolerance-tested fixtures; full SVD
  remains unavailable here by design (use PCA for dense full-rank work).
- **Explained variance is uncentered** → defined from the transformed data, matching the
  reference; documented.
- **Sign ambiguity** → the kernel's fixed convention.
- **Sparse product performance** → the sparse path is the point; no densification.

## Migration Plan

Additive: new estimator in `sciencekit_decomposition`. No existing API changes.

## Open Questions

None blocking. ARPACK and an adaptive rank selector are future changes.
