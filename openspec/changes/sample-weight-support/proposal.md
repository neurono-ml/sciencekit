# Proposal

## Why

The parity audit found that no public fit signature accepts per-sample weights: the
decision tree computes weighted impurities internally but never exposes the parameter, and
the SGD family synthesizes a vector of ones inside the epoch loop. scikit-learn supports
`sample_weight` on most estimators, and the parity matrix cannot claim an estimator as
fully implemented while the weights are unavailable. Without a shared contract, the 150
upcoming algorithm changes would each invent their own weight plumbing.

## What Changes

- Add an optional `sample_weight` to `fit`, `partial_fit` and `fit_streaming` for every
  estimator whose mathematics admits weights: the scalers and imputers that aggregate
  (StandardScaler, MinMaxScaler, RobustScaler, SimpleImputer), the linear family
  (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGD), the SVM family,
  the tree and ensemble families, KMeans and MiniBatchKMeans, and the KNN imputer's donor
  weighting.
- Extend the streaming batches (`SKDataBatch`, `SKSupervisedBatch`) with an optional
  weight vector, so weighted streaming fits are possible; the double-buffer guard accounts
  for the extra array.
- Remove the synthesized `vec![F::one()]` inside SGD: weights become real input, not a
  placeholder.
- Define the semantics: weights are non-negative and finite; zero-weight rows are dropped
  from the fit statistics; class weights compose multiplicatively with sample weights;
  weighted metrics (`sk_weighted_average` and the scorers) consume the same weights.
- Validation: negative, non-finite, or length-mismatched weights fail with the central
  taxonomy; the error names the offending row range.
- Explicit non-goals: per-group weights, case weights distinct from sample weights,
  changing unweighted results (a missing weight is exactly one), and the estimators whose
  mathematics does not admit weights (they refuse with a typed `UnsupportedOperation`).

## Capabilities

### New Capabilities

- `sample-weights`: the per-sample weight contract, its validation, its composition with
  class weights, and the weighted-statistics reuse across estimators and metrics.

### Modified Capabilities

None. The change is additive to the public signatures.

## Impact

- `sciencekit_common`: weighted batch types and validation.
- Every estimator crate listed above: the parameter and its propagation into the existing
  kernels (the tree splitter and the coordinate-descent accumulator already compute
  weighted statistics internally).
- `sciencekit_metrics`: weighted scoring entry points wired to the same weights.
- The DAG entry exists; this change fills it.
