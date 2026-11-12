# Design

## Context

See `proposal.md` — Why. `sciencekit_cluster` is new. `sciencekit_common` supplies
`SKUnsupervisedFit`, `SKClustererPredictor` (from `clustering-contracts`), builder state,
data views and observability. `sciencekit_math` supplies dense/sparse products and
reductions. The assignment contract returns the typed `SKClusterLabel` from
`clustering-contracts`; KMeans only produces `Cluster` variants.

## Goals / Non-Goals

**Goals:**
- Exact Lloyd/Elkan KMeans with deterministic initialization and empty-cluster repair.
- Dense and sparse parity without densification.
- A model reusable by MiniBatchKMeans and DBSCAN.

**Non-Goals:**
- Streaming/`partial_fit` (W5.2), missing values, GPU, `memmap2`, export.
- Other metrics (cosine, Manhattan); KMeans is Euclidean in this change.
- Silhouette scoring (later metrics work).

## Decisions

### Decision 1 — New `sciencekit_cluster` crate

The crate owns KMeans, MiniBatchKMeans and DBSCAN. KMeans stores dense centers and the
typed training assignments; sparse input is read through the sparse norm-expansion
distance. A `SKClusterModel` type is the fitted model; no configured-estimator prediction
exists.

### Decision 2 — Squared Euclidean assignment via norm expansion

Assignment uses `||x − c||² = ||x||² − 2 x·c + ||c||²`; sparse rows use their stored
entries and precomputed norms. The center matrix is dense (`k × d`), which is the expected
memory profile.

### Decision 3 — k-means++ with seeded D² sampling

The first center is a uniformly chosen row; each next center is sampled with probability
proportional to `D(x)²`, the squared distance to the nearest chosen center. The sampler
uses a local SplitMix64 generator derived from `random_seed` and the initialization index;
a future shared generator in `sciencekit_common` can replace it.

### Decision 4 — Lloyd and Elkan

Lloyd alternates assignment and mean update. Elkan maintains:
- center-to-center distances and half-distances;
- per-row upper bound to its assigned center and lower bound to every other center;
- the triangle-inequality tests that skip distance evaluations.

Elkan runs on dense Euclidean data; `Auto` selects Elkan for dense and Lloyd for sparse
(where bounds are still valid but the bookkeeping is not worth it). Both paths produce the
same labels and centers within tolerance; a test asserts equality.

### Decision 5 — Best-of-initializations

`number_of_initializations` runs independent seeded initializations. They are independent
and run in parallel; the model with the smallest inertia wins, ties by the earliest index.
Each initialization has its own seed derived from `random_seed`.

### Decision 6 — Empty-cluster repair

If a center receives no rows, the point with the largest distance to its assigned center is
moved to the empty center. The point choice is deterministic: largest distance, then
smallest row index.

### Decision 7 — Convergence semantics

The update stops when the maximum center movement is at or below
`tolerance * mean_squared_norm(centers)` (the scikit-learn tolerance semantics), or when
the iteration budget is exhausted. The model reports `converged()` and
`iterations_taken()`; exhaustion is not an error because KMeans is a heuristic
optimization.

### Decision 8 — Model API

```rust
pub struct SKKMeansModel<F: SKFloat> {
    centers: Array2<F>,
    labels: Array1<SKClusterLabel>,
    inertia: F,
    iterations: usize,
    converged: bool,
    backend: SKBackendKind,
}
```

Accessors: `cluster_centers`, `labels`, `inertia`, `iterations_taken`, `converged`,
`number_of_clusters`, `backend`. `predict_assignments`, `transform` and `fit_predict` are
exposed; `fit` returns the model through `SKUnsupervisedFit`.

### Decision 9 — Validation and observability

Builder validation: clusters > 0, clusters ≤ samples at fit, iterations > 0, tolerance ≥ 0,
initialization count > 0. Fit/assignment validate finiteness and feature counts. Spans
record the configured backend; fit transfers it to the model.

### Decision 10 — File layout

```
crates/sciencekit_cluster/src/
  lib.rs
  kmeans/{mod.rs, builder.rs, config.rs, init.rs, lloyd.rs, elkan.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Elkan bookkeeping is complex** → a dedicated equality test against Lloyd and a
  dense-only policy reduce the risk.
- **Empty clusters can destabilize** → deterministic repair and tests.
- **KMeans is sensitive to scaling** → documented; users can compose a scaler.
- **Sparse assignment is slower than dense** → the sparse path is still zero-copy and
  avoids densification.
- **No streaming** → MiniBatchKMeans (W5.2) is the streaming path.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. Other metrics and GPU are future changes.
