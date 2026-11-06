# Proposal

## Why

Wave 5 shipped KMeans, MiniBatchKMeans, DBSCAN and their typed assignments, but the
metrics crate has no way to judge a clustering. Silhouette, adjusted Rand index, normalized
mutual information, homogeneity/completeness/V-measure, Davies-Bouldin and
Calinski-Harabasz are the standard tools, and the existing `SKUnsupervisedScorer` contract
has no concrete implementation. This change closes that gap.

## What Changes

- Add `sk_silhouette_samples` and `sk_silhouette_score` over features and typed
  assignments, excluding noise rows.
- Add pairwise-labeling metrics: adjusted Rand index, normalized mutual information
  (arithmetic, geometric, max), homogeneity, completeness and V-measure.
- Add centroid/distance-based clustering scores: Davies-Bouldin and Calinski-Harabasz.
- Add `SKSilhouetteScorer` implementing `SKUnsupervisedScorer` and filtering the typed
  `Noise` variant.
- Dense and sparse inputs through the pairwise-distance dispatcher; no densification.
- Validation: empty input, assignment length mismatch, fewer than two clusters, a single
  cluster for silhouette, noise-only input and non-finite values.
- Explicit non-goals: biclustering metrics, gap statistic, stability selection, `memmap2`
  and export.
- Companion tests: hand-computed two-cluster fixtures, noise exclusion, dense/sparse
  equivalence, large data, concurrency and scorer equivalence.

No existing metric behavior changes and no breaking changes.

## Capabilities

### New Capabilities

- `clustering-metrics`: silhouette, adjusted Rand index, normalized mutual information,
  homogeneity/completeness/V-measure, Davies-Bouldin, Calinski-Harabasz and the silhouette
  scorer.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_metrics` gains clustering modules and scorers.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** gives Wave 5 clusterers a real evaluation surface and lets model
  selection choose a cluster count by silhouette.
- **Acceptance criteria:** small and large data, dense/sparse, concurrency, companion
  `*_tests.rs` modules.
