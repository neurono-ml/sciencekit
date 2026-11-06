# Proposal

## Why

KMeans is the first unsupervised estimator in the roadmap and the baseline every clustering
user expects. The library has no cluster crate, no centroid initialization and no
assignment contract yet; the latter is supplied by `clustering-contracts`. KMeans is also
the first consumer of the Elkan triangle-inequality acceleration and a reference for the
streaming MiniBatch variant.

## What Changes

- New sub-crate `sciencekit_cluster` with `SKKMeans<F>` and a distinct fitted model.
- Builder-only surface: `number_of_clusters` (default 8), `initialization`
  (`KMeansPlusPlus` default, `Random`), `number_of_initializations` (default 10 for
  random, 1 for k-means++), `maximum_number_of_iterations` (default 300), `tolerance`
  (default 1e-4), `algorithm` (`Auto` default, `Lloyd`, `Elkan`), `random_seed` (default
  0), `backend` (default `Faer`) and `execution_mode` (default `Automatic`).
- Fit supports dense and sparse features, k-means++ and random initialization, Lloyd and
  Elkan assignment, deterministic empty-cluster repair and the best-of-`n_init` inertia
  selection.
- The fitted model exposes cluster centers, labels, inertia, the number of iterations and
  the convergence flag, and implements `SKUnsupervisedFit` plus `SKClustererPredictor`.
- `predict`, `fit_predict` and `transform` (distances to centers) are available; labels are
  typed `SKClusterLabel::Cluster` variants in `0..number_of_clusters`; KMeans never emits
  `Noise`.
- Validation: empty input, zero clusters, clusters greater than samples, non-finite values,
  invalid tolerance/iteration/initialization parameters and feature-count mismatch.
- Explicit non-goals: streaming/`partial_fit` (W5.2), missing values, GPU, `memmap2` and
  export (Wave 7).
- Companion tests: hand-checkable two-cluster fixtures, k-means++ coverage, Lloyd/Elkan
  equivalence, sparse/dense equivalence, empty-cluster repair, determinism, concurrency,
  large data and inertia/metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `k-means`: `SKKMeans` — builder surface, initialization, Lloyd/Elkan assignment,
  empty-cluster handling, center storage, assignment prediction, representation parity,
  determinism and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_cluster` workspace member; root `Cargo.toml`
  membership.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  `sciencekit_metrics` as a dev-dependency.
- **Downstream:** unblocks MiniBatchKMeans and DBSCAN in the same crate; gives later model
  selection a clustering estimator.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
