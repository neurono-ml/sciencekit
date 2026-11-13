# Proposal

## Why

`outlier-detection` shipped `SKIsolationForest` and closed Wave 5 with a single detector,
but the public catalog still lists "Anomaly detection (LOF, One-Class SVM)" as Planned
because two canonical scikit-learn detectors have no change, no DAG entry and no code:
`LocalOutlierFactor` and `OneClassSVM`.

Both are now unblocked by foundations that did not exist when the catalog was written:

- `kd-tree-and-ball-tree` supplies exact dense spatial indexes with the shared
  `SKNearestNeighborsSearcher` contract, and its brute-force searcher accepts sparse and
  mixed representations without densifying.
- `svm-svc-svr` supplies the kernel catalogue and the SMO solver primitives.

LOF is the standard density-based detector for finding local anomalies that a global
method such as Isolation Forest misses. One-Class SVM is the standard boundary-based
detector for novelty detection with a small, explicit kernel surface. This change turns the
catalog row into a concrete, testable plan.

## What Changes

- Add `SKLocalOutlierFactor<F>` to `sciencekit_outlier`, constructed only through
  `SKLocalOutlierFactorBuilder`, exposing `nearest_neighbors_count` (default 20),
  `contamination` (`Auto` default or a fraction in `(0, 0.5]`), `novelty` (default false),
  `metric` (default Euclidean), `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`).
- Implement the LOF geometry explicitly: the core distance to the k-th neighbour, the
  reachability distance `max(core_distance, distance)`, the local reachability density
  (inverse mean reachability distance to the k neighbours) and the local outlier factor
  (mean neighbour density over the point's density).
- Expose `negative_outlier_factor` and `fit_predict` for the fit-only mode
  (`novelty = false`), and `score_samples`, `decision_function` and `predict` for
  `novelty = true`; the fit-only mode refuses prediction on new data with a typed
  `UnsupportedOperation` that points at `novelty(true)`.
- Add `SKOneClassSVM<F>` with `nu` in `(0, 1]` (default 0.5), the linear, polynomial,
  radial-basis-function and sigmoid kernels from `sciencekit_math::kernels` (precomputed
  excluded because the estimator never sees a precomputed training kernel), `gamma`,
  `degree`, `coef0`, `tolerance`, `maximum_number_of_iterations`, `cache_size`,
  `shrinking`, `backend` and `execution_mode`.
- Reuse the `sciencekit_svm` SMO primitives (working-set selection, gradient/step,
  shrinking, stopping criterion and kernel cache) to solve the one-class dual; fit produces
  support vectors, dual coefficients and the bias `rho`.
- Both detectors implement `SKOutlierPredictor<F>`, returning `SKOutlierLabel` and the
  `score_samples`/`decision_function` pair with the contamination-derived offset.
- Representation parity: dense fits use the KD/ball tree index, sparse fits use the
  brute-force searcher, and no input is densified. The data-view decision is recorded in
  `design.md` per the `data-view-boundary` spec.
- Both fitted models implement `SKExportable`/`SKImportable`; no typed refusal is planned
  (see `design.md`, Decision 7).
- Validation: empty input, `nearest_neighbors_count` outside `1..=n`, contamination
  outside `(0, 0.5]`, `nu` outside `(0, 1]`, invalid kernel parameters, non-finite values,
  feature-count mismatch, prediction before fit and prediction on new data in fit-only
  mode.
- Explicit non-goals: missing values, streaming/`partial_fit`, GPU execution, `memmap2`,
  precomputed kernels, multi-class novelty, local outlier probability (LoOP) and
  incremental LOF.

No existing estimator API changes beyond the additive estimators.

## Capabilities

### New Capabilities

- `outlier-lof-one-class`: `SKLocalOutlierFactor` and `SKOneClassSVM` — builder surfaces,
  LOF density geometry, one-class SMO fitting, kernel selection, novelty semantics,
  representation parity, exporter/importer and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_outlier` gains `local_outlier_factor/` and
  `one_class_svm/` folder modules plus acceptance tests; `sciencekit_common` is unchanged
  because `SKOutlierPredictor`, `SKOutlierLabel` and the export traits already exist.
- **APIs:** additive estimators; `sciencekit_outlier` gains dependencies on
  `sciencekit_neighbors`, `sciencekit_svm`, `sciencekit_math` and `sprs`.
- **Dependencies:** `sciencekit_common`, `sciencekit_neighbors`, `sciencekit_svm`,
  `sciencekit_math`, `ndarray`, `sprs`, `rayon`; `sciencekit_metrics` stays a
  dev-dependency for acceptance metrics.
- **Downstream:** completes the outlier block named in the catalog; the models become
  available to the export retrofit and the umbrella crate.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  dense/sparse parity, f32/f64, export/import round-trip, companion `*_tests.rs` modules.
