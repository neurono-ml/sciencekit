# Design

## Context

See `proposal.md` — Why. `sciencekit_impute` already owns `SKSimpleImputer` and
`SKMissingIndicator`; `sciencekit_math` exposes `sk_nan_euclidean_distance_matrix`;
`sciencekit_common` provides the marker conventions, builder state, data views and
observability. The NaN-aware distance is dense-only and is not a metric, so a spatial
index cannot prune it soundly.

## Goals / Non-Goals

**Goals:**
- Exact scikit-learn parity for the `nan_euclidean` donor semantics.
- Bounded memory and deterministic parallel transform.
- Reuse the existing marker and indicator conventions rather than inventing new ones.

**Non-Goals:**
- Sparse input and a sparse missing marker.
- A spatial index for `nan_euclidean`.
- `partial_fit`/`fit_streaming`, `memmap2` backing and model export.

## Decisions

### Decision 1 — Brute force is the correct algorithm, not a placeholder

`nan_euclidean` compares only co-observed features and rescales by
`total_features / present_features`; the triangle inequality fails when the observed
sets differ. Ball-tree and KD-tree pruning are therefore unsound. scikit-learn's
`KNNImputer` deliberately uses `pairwise_distances_chunked` with `nan_euclidean`, and its
tree `valid_metrics` do not include `nan_euclidean`. This change follows the reference:
the exact algorithm is the chunked brute-force distance, and the wave-plan wording that
described the imputer as tree-backed is superseded.

Future accelerators (a lattice/pattern index with partial-distance pruning, or an
approximate index) can be added without changing the observable result. They are not part
of this change.

### Decision 2 — Store the training design, mask and valid columns

Fit stores the dense training matrix, its missingness mask, the set of columns with at
least one observed value and the observed column means used by the fallback. All-missing
columns are excluded by default; `keep_empty_features` keeps them at zero. The model is
immutable and `Send + Sync`.

### Decision 3 — Chunked distance computation with per-column donors

Transform processes receiver rows in fixed-size chunks. For each chunk it computes the
`chunk × n_training` NaN-aware distance matrix through
`sk_nan_euclidean_distance_matrix`, then per column:

1. restrict to training rows that observed that column (precomputed indices);
2. select the `k` smallest distances with `select_nth_unstable_by`;
3. if no selected distance is finite, use the column mean;
4. otherwise compute uniform or inverse-distance weights and average the donor values.

Chunk width is derived from a fixed memory budget, so the full distance matrix is never
materialized. Receiver rows are independent; rayon parallelizes chunks or rows and
collects in order.

Alternative considered: a per-pair loop with early missing checks (simpler but slower and
duplicates the scalar kernel).

### Decision 4 — Distance weights and fallback

If any selected donor is at distance zero, only zero-distance donors contribute (the
scikit-learn convention), avoiding infinite weights. Otherwise the weight is
`1 / distance`. When every candidate distance is NaN, the column mean over observed
training values is used; this is the documented fallback, not a silent sentinel.

### Decision 5 — Marker and indicator reuse

Missing detection uses the same predicate as `SKSimpleImputer`: `NaN` tests `is_nan`, any
other scalar tests equality. `add_indicator` composes the existing `SKMissingIndicator`
scope: it records the columns missing at fit, stacks a binary mask after the imputed
features, and tolerates missingness unseen at fit without expanding the mask.

### Decision 6 — Dense-only exception

A CSR implicit zero is an observed zero, so there is no missing marker in the sparse
representation. The imputer therefore rejects sparse input with
`UnsupportedRepresentation` and points at `SKSimpleImputer` for sparse data. This is the
documented dense-only exception in `representation-parity`: the imputer's mathematics
(and the marker contract) does not tolerate a sparse missing representation.

### Decision 7 — Validation and observability

Builder: `backend` default `Faer`, `execution_mode` default `Automatic`; fit transfers
both into the model. `fit`/`transform` run in the shared operation span and record the
backend. Errors: `EmptyInput`, `InvalidHyperparameter` (neighbour count, marker value),
`ShapeMismatch`, `NonFiniteValue`, `UnsupportedRepresentation` (sparse).

### Decision 8 — File layout

```
crates/sciencekit_impute/src/knn/
  mod.rs
  builder.rs
  core_implementation.rs
  model.rs
  fit.rs
  transform.rs
  donors.rs
  distance_block.rs
  *_tests.rs
  acceptance_tests/mod.rs
```

`mod.rs` is a pure dispatcher; companion `*_tests.rs` modules are declared at the end.

## Risks / Trade-offs

- **`O(n_receivers · n_training)` time** → inherent to exact `nan_euclidean`; chunking
  bounds memory, rayon bounds wall-clock.
- **All-NaN distances** → explicit column-mean fallback, tested on sparse observation
  patterns.
- **Zero-distance duplicates** → zero-distance weight convention, tested.
- **No sparse support** → documented dense-only exception; `SKSimpleImputer` remains the
  sparse-capable option.
- **Chunk heuristic can be conservative** → exposed as a batch/memory budget and tested
  on large input.

## Migration Plan

Additive: a new estimator in `sciencekit_impute`. No existing API changes.

## Open Questions

None blocking. An exact incomplete-kNN index (lattice/pattern buckets with
partial-distance pruning) is a future accelerator and would keep the same observable
results.
