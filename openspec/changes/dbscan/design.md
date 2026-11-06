# Design

## Context

See `proposal.md` — Why. `sciencekit_neighbors` supplies the dense tree indexes and the
`SKNearestNeighborsSearcher` radius query; the brute-force searcher covers sparse/mixed
data. `sciencekit_common` supplies the cluster assignment contract (from
`clustering-contracts`). DBSCAN is the first consumer of the radius query.

## Goals / Non-Goals

**Goals:**
- Exact density semantics matching the reference.
- Deterministic labels independent of thread count.
- Sparse support through brute force without densification.
- A useful assignment extension for new points.

**Non-Goals:**
- HDBSCAN/OPTICS, missing values, streaming, GPU, `memmap2`, export.

## Decisions

### Decision 1 — Region query through the shared searcher

`Auto` chooses a tree for dense Euclidean/Manhattan/Chebyshev data and brute force for
sparse or mixed data, mirroring KNN. The tree's `radius_neighbors` returns the
neighbourhood; brute force uses the mixed distance dispatcher in chunks. The point itself
is included in its neighbourhood, matching scikit-learn's `min_samples` semantics.

### Decision 2 — Standard DBSCAN expansion

For each unvisited point in training-index order:

1. query its neighbourhood;
2. if it has fewer than `minimum_samples` points, mark it noise (it may later become a
   border point);
3. otherwise start a new cluster with the next id and expand: every unvisited neighbour
   becomes part of the cluster; if that neighbour is core, its neighbourhood is added to
   the expansion queue;
4. border points receive the cluster that reaches them first.

The order is deterministic because the outer loop and the queue use training indices.

### Decision 3 — Cluster ids

Cluster ids are assigned sequentially as core points seed components, starting at zero.
Noise is the explicit `SKClusterLabel::Noise` variant. The smallest training index in a
component therefore determines its id.

### Decision 4 — Assignment extension

`predict_assignments` finds the nearest core sample (through the same searcher) and assigns
its cluster if the distance is within `eps`; otherwise it returns
`SKClusterLabel::Noise`. This is an
extension beyond scikit-learn (which has no DBSCAN `predict`) and is documented as a
nearest-core heuristic, not a re-clustering.

### Decision 5 — Representation parity

Dense trees are used when possible; sparse data goes through the brute-force searcher over
the stored CSR rows. No path densifies a sparse matrix. The assignments are the typed
`SKClusterLabel` from the shared clustering contract.

### Decision 6 — Model state and observability

The model stores labels, core-sample indices, cluster count and `eps`. Fit and assignment
spans record the backend. Errors: `EmptyInput`, `InvalidHyperparameter`, `ShapeMismatch`,
`NonFiniteValue`, `UnsupportedRepresentation` and the tree/searcher errors converted into
the central taxonomy.

### Decision 7 — File layout

```
crates/sciencekit_cluster/src/dbscan/
  mod.rs, builder.rs, config.rs, region_query.rs, expansion.rs, model.rs,
  fit.rs, predict.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Worst-case `O(n²)` region queries** → the tree accelerates dense metric data; sparse
  data is inherently brute force in this version.
- **Border-point assignment order** → deterministic training-index order; documented.
- **`eps` is scale-sensitive** → documented; users scale features first.
- **Prediction is a heuristic** → documented as nearest-core, not a re-clustering.
- **No HDBSCAN** → future change.

## Migration Plan

Additive: new estimator in `sciencekit_cluster`. No existing API changes.

## Open Questions

None blocking. HDBSCAN/OPTICS and a sparse spatial index are future changes.
