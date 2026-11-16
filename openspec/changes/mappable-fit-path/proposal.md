# Proposal

## Why

PRD §4.4 promises that every algorithm declares its access pattern and implements the
matching source trait, and it names KMeans, KNN, DBSCAN and the decision trees as the
mappable (`RandomAccess`) examples. Today **no estimator consumes `SKMappableSource`**: the
trait exists in `sciencekit_common::batching` and concrete mapped sources exist in
`sciencekit_interop::memmap`, but every `fit` only accepts an in-memory `SKDataView`. The
conformance audit (`sklearn-parity-matrix/conformance-audit.md`) records this as the
largest gap between the out-of-core promise and the implementation.

## What Changes

- Add a mappable fit entry point for every estimator whose declared access pattern is
  `RandomAccess`: `SKKNeighborsClassifier`, `SKKNeighborsRegressor`, `SKKNNImputer`,
  `SKKDTree`, `SKBallTree`, `SKDBSCAN`, `SKDecisionTreeClassifier`,
  `SKDecisionTreeRegressor`, `SKRandomForest*`, `SKBagging*`, `SKIsolationForest`.
- Extend the batching contracts with a reusable validation helper so an estimator rejects a
  non-mappable source before reading any row, with a structured error.
- Define model ownership over mapped storage: a model holds a shared handle to the mapped
  source when the fitted state needs the training design (neighbors, KNN imputer), and
  owns only its compact tensors when it does not (trees, forests, DBSCAN labels); export
  keeps materializing owned tensors so persistence is unaffected.
- Guarantee result parity within the documented tolerance between the in-memory and the
  mapped fit of the same logical data, on dense and CSR layouts.
- Record the declared pattern in the operation span (already required by
  `access-pattern-contract`).
- Explicit non-goals: streaming (`reopenable-fit-path` covers iterative estimators), GPU
  backends, new mapped formats, changing the in-memory results.

## Capabilities

### New Capabilities

- `mappable-fit`: the mappable-source fit contract for random-access estimators, its
  ownership rule, its validation point and its parity guarantee.

### Modified Capabilities

None.

## Impact

- `sciencekit_common`: validation helper and the mapped-source hand-off.
- `sciencekit_neighbors`, `sciencekit_impute`, `sciencekit_cluster`, `sciencekit_tree`,
  `sciencekit_ensemble`, `sciencekit_outlier`: the mappable fit paths.
- `sciencekit_interop`: the mapped sources become first-class fit inputs.
- The DAG entry exists; this change fills it.
