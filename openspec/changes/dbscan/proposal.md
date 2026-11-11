# Proposal

## Why

KMeans assumes convex, similarly sized clusters. DBSCAN finds arbitrary-shaped clusters
from density, labels low-density points as noise and does not need the cluster count up
front. It is the canonical density baseline and the first real consumer of the spatial
index from Wave 3 (W3.2) through its region query.

## What Changes

- New `SKDBSCAN<F>` in `sciencekit_cluster`, producing a fitted model with labels and core
  sample information.
- Builder-only surface: `eps` (default 0.5), `minimum_samples` (default 5), `metric`
  (`Euclidean` default, `Manhattan`, `Chebyshev`, `Cosine`), `algorithm` (`Auto` default, `KdTree`,
  `BallTree`, `BruteForce`), `leaf_size` (default 40), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`).
- Fit uses a region query: trees for dense metric-compatible data, brute force for sparse
  or mixed data; dense and sparse inputs are both accepted and sparse is never densified.
- Labeling: core points, density-reachable border points and the explicit
  `SKClusterLabel::Noise` variant, with deterministic label order (the smallest training
  index in each component seeds the cluster id).
- The model exposes typed assignments, core sample indices, cluster count and the fitted
  `eps`, and implements `SKUnsupervisedFit` plus `SKClustererPredictor`. `fit_predict` is
  provided; `predict_clusters` assigns a new point to the nearest core sample's cluster when it lies
  within `eps`, otherwise `SKClusterLabel::Noise`, and documents this extension.
- Validation: empty input, non-positive `eps`, zero `minimum_samples`, feature-count
  mismatch, non-finite values and an unsupported metric/algorithm pair.
- Explicit non-goals: HDBSCAN/OPTICS, streaming, missing values, GPU, `memmap2` and
  export (Wave 7).
- Companion tests: hand-checkable clusters and noise, core/border semantics, tree vs
  brute-force equality, sparse/dense equivalence, deterministic labels, concurrency,
  large data and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `dbscan`: `SKDBSCAN` — region query, core/border/noise semantics, deterministic label
  assignment, the prediction extension, representation parity and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_cluster` gains the DBSCAN module and model.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_neighbors` (tree
  and searcher), `ndarray`, `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** gives the library a density clusterer and validates the spatial-index
  region query outside KNN.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
