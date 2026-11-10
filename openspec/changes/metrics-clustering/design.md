# Design

## Context

See `proposal.md` — Why. `sciencekit_metrics` already owns the label space and the
regression/classification families. `sciencekit_common` owns `SKUnsupervisedScorer` and the
typed `SKClusterLabel` from `clustering-contracts`. `sciencekit_math` owns the dense/sparse
pairwise distances. The metrics crate can depend on `sciencekit_math` for silhouette.

## Goals / Non-Goals

**Goals:**
- Standard clustering metrics with clear orientation (higher/lower is better).
- Noise-aware silhouette and centroid scores.
- Dense/sparse parity without densification.

**Non-Goals:**
- Biclustering metrics, gap statistic, stability selection.
- Approximate silhouette on very large data (future sampling accelerator).
- `memmap2` and export.

## Decisions

### Decision 1 — Silhouette from pairwise distances

For each non-noise sample, compute `a` (mean distance to its own cluster) and `b` (minimum
mean distance to another cluster); the sample value is `(b − a) / max(a, b)`. Distances
come from `sk_pairwise_distance_matrix`, chunked by row, so memory is `O(n)` per row rather
than a full `n × n` matrix. Rows are independent and parallel.

### Decision 2 — Pairwise labeling metrics

ARI, NMI and homogeneity/completeness/V-measure operate on two `i64` label vectors. Build a
contingency table over the union label space, then compute entropies and mutual
information. Labels are canonicalized; noise is just another label for pairwise metrics
(the caller decides whether to filter it).

### Decision 3 — Centroid scores

Davies-Bouldin uses cluster centroids and within-cluster scatter; Calinski-Harabasz uses
the between/within dispersion ratio. Both accept dense and sparse features; centroids are
dense and sparse products avoid densifying the input. Noise rows are excluded.

### Decision 4 — Silhouette scorer

`SKSilhouetteScorer` implements `SKUnsupervisedScorer<F, M>` for a model implementing
`SKClustererPredictor<F>`; it delegates to the free function, which excludes `Noise`. The
scorer is the first concrete unsupervised scorer.

### Decision 5 — Validation and observability

Validate row counts, finiteness, label space size and non-noise cluster count. All
functions are pure and thread-safe; concurrent tests assert equality. No backend is carried
because these are standalone functions (the backend rule binds estimators).

### Decision 6 — File layout

```
crates/sciencekit_metrics/src/clustering/
  mod.rs, silhouette.rs, adjusted_rand.rs, mutual_information.rs,
  homogeneity.rs, davies_bouldin.rs, calinski_harabasz.rs, *_tests.rs
crates/sciencekit_metrics/src/scorers/clustering.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Silhouette is `O(n²)` time** → chunked and parallel; a sampling accelerator is future
  work. Document the cost.
- **Noise handling differs per metric** → silhouette/centroid scores exclude it; pairwise
  metrics treat it as a label and document that.
- **Sparse centroid scores can densify centroids** → centroids are `O(k · d)`, much smaller
  than the data; the input itself is never densified.
- **Label permutation invariance** → pairwise metrics are permutation-invariant; tests
  cover it.

## Migration Plan

Additive: new modules, scorer and re-exports. No existing metric changes.

## Open Questions

None blocking. Approximate silhouette and biclustering metrics are future changes.
