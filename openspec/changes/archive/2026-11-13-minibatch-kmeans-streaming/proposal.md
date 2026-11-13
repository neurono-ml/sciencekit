# Proposal

## Why

Lloyd/Elkan KMeans needs the full training matrix and re-scans it every iteration, which
does not scale to data that does not fit in memory. MiniBatchKMeans updates centers from
small batches with per-center learning rates, supports `partial_fit` and the streaming
executor, and is the PRD's out-of-core clustering differentiator.

## What Changes

- `SKMiniBatchKMeans<F>` in `sciencekit_cluster`, reusing the KMeans center/assignment
  surfaces and the `clustering-contracts` predictor.
- Builder-only surface: `number_of_clusters` (default 8), `initialization`
  (`KMeansPlusPlus` default, `Random`), `number_of_initializations` (default 3),
  `batch_size` (default 1024), `initialization_batch_size` (default 3 × `batch_size`),
  `maximum_number_of_iterations` (default 100), `maximum_number_of_epochs` (default 1 for
  `fit_streaming`), `tolerance` (default 0), `reassignment_ratio` (default 0.01),
  `random_seed` (default 0), `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`).
- `partial_fit` updates centers from a batch with the per-center learning rate
  `1 / (count[center] + 1)`; `fit_streaming` drives an `SKSourceProvider` through the shared
  prefetch executor and supports multiple epochs when the provider is reopenable.
- The model exposes centers, labels, inertia, per-center counts and the batch/epoch
  progress, and implements `SKUnsupervisedFit` plus `SKClustererPredictor`; `predict`,
  `fit_predict` and `transform` are available.
- Sparse batches are accepted and never densified; the centers stay dense.
- Validation: empty batch, zero clusters, batch smaller than the cluster count, invalid
  initialization size, non-finite values, invalid reassignment ratio and feature-count
  mismatch.
- Explicit non-goals: Elkan on streaming batches, missing values, GPU, `memmap2` and
  export (Wave 7).
- Companion tests: hand-checkable batch update, per-center learning-rate decay, streaming
  equals repeated `partial_fit`, reopenable multi-epoch fit, sparse/dense equivalence,
  determinism, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `mini-batch-k-means`: `SKMiniBatchKMeans` — batch update rule, per-center learning
  rates, initialization, streaming fit, epoch handling, center storage, assignment
  prediction, representation parity and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_cluster` gains the miniature-batch module and model.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common` (streaming sources and executor), `sciencekit_math`,
  `ndarray`, `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** gives pipelines and out-of-core workflows a clustering estimator and
  exercises the reopenable streaming-provider path.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
