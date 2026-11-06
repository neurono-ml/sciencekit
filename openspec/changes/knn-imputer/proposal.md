# Proposal

## Why

`sciencekit_impute` currently fills missing cells with per-column statistics. The
nearest-neighbor imputer is the multivariate alternative: a cell is filled from the
observed values of the training rows that are closest on the co-observed features. The
`nan_euclidean` kernel and the evaluation metrics are now merged, so the imputer can be
built on the exact algorithm scikit-learn uses instead of waiting for a sparse index.

## What Changes

- `SKKNNImputer<F>` in `sciencekit_impute`, with a distinct fitted model.
- Builder-only surface: `missing_values` (default `NaN`), `nearest_neighbors_count`
  (default 5), `weights` (`Uniform` default, `Distance`), `add_indicator` (default false),
  `keep_empty_features` (default false), `backend` (default `Faer`) and `execution_mode`
  (default `Automatic`).
- Exact transform: chunked `nan_euclidean` distances from the rows that need imputation to
  the training rows; per column, donors with an observed value for that column; `k`
  nearest via partial selection; uniform or inverse-distance weighted average; column-mean
  fallback when every candidate distance is NaN.
- Missing-marker semantics shared with `SKSimpleImputer` and `SKMissingIndicator`: `NaN`
  tests `is_nan`, any other scalar tests equality, all-missing training columns are
  dropped by default (kept at zero when `keep_empty_features`), and `add_indicator`
  stacks the fitted missingness mask.
- Dense input only: the NaN-aware distance has no sparse missing marker (an implicit CSR
  zero is an observed zero), so a sparse input is rejected with
  `UnsupportedRepresentation` and a suggestion to use `SKSimpleImputer`. This is a
  documented dense-only exception; the imputer's public surface does not claim sparse
  parity.
- Bounded memory: distances are computed in fixed-size chunks, and rows that need
  imputation are processed independently and in parallel.
- Validation: empty input, zero or excessive neighbour count, feature-count mismatch,
  non-finite values other than the configured marker, and empty observation sets.
- Explicit non-goals: `partial_fit`/`fit_streaming` (the imputer retains the training
  design by definition; incremental storage is a later change), `memmap2` backing and
  model export (Wave 7).
- Companion tests: hand-checked imputation, missing patterns, weighted/unweighted donors,
  mean fallback, indicator composition, large data, concurrency and metric production.

No existing imputer API changes and no breaking changes.

## Capabilities

### New Capabilities

- `nearest-neighbor-imputation`: `SKKNNImputer` — builder surface, marker semantics,
  per-column donor selection with `nan_euclidean`, weighted averaging, fallback behavior,
  indicator composition, bounded-memory transform, dense-only contract and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_impute` gains the `knn` module and model; no new crate.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math` (`sk_nan_euclidean_distance_matrix`),
  `sciencekit_metrics` as a dev-dependency for acceptance tests.
- **Downstream:** gives pipelines a multivariate imputer and exercises the NaN-aware
  distance in a real estimator; `SKSimpleImputer` remains the sparse-capable option.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent transform,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Model export is
  deferred to Wave 7 and recorded in the PRD.
