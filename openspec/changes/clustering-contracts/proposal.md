# Proposal

## Why

The library is about to ship its first clusterers, but the common contracts have no way to
represent cluster assignments: the unsupervised scorer is bound to `SKRegressorPredictor`
and rounds float responses into `usize` indices, which cannot represent DBSCAN's noise
label and makes clustering evaluation an accident of the regressor contract. Clustering
needs one honest assignment surface before KMeans, MiniBatchKMeans and DBSCAN are built.

## What Changes

- Add `SKClustererPredictor<F>` to `sciencekit_common`: a model-side contract returning
  `Array1<i64>` cluster assignments, with `-1` reserved for noise.
- Change `SKUnsupervisedScorer` to be generic over `SKClustererPredictor<F>` instead of
  `SKRegressorPredictor<F>`; its pure form accepts `ArrayView1<i64>` assignments and its
  convenient form obtains assignments directly from the model.
- Define how scorers treat noise assignments: a scorer SHALL either exclude `-1` rows or
  reject them, and the contract SHALL make that choice explicit per scorer.
- Keep `SKUnsupervisedFit` as the fitting contract; clusterers implement it and the new
  predictor on their fitted model.
- No concrete clusterer, metric or KMeans code in this change.

This is a **BREAKING** change to the `SKUnsupervisedScorer` trait bound, acceptable while
the library is pre-1.0 and no concrete unsupervised scorer has shipped.

## Capabilities

### New Capabilities

- `clustering-contracts`: the cluster assignment predictor contract, its noise label and
  the assignment shape/scalar guarantees.

### Modified Capabilities

- `scoring-contracts`: the unsupervised scorer's pure and convenient forms now operate on
  cluster assignments through `SKClustererPredictor`, not on regressor responses.

## Impact

- **Code:** `sciencekit_common::fit_traits` (new trait and re-export) and
  `sciencekit_common::scorer_traits` (bound and assignment type).
- **APIs:** **BREAKING** for any external implementor of `SKUnsupervisedScorer`; no
  concrete implementation exists in the workspace yet.
- **Downstream:** unblocks `SKKMeans`, `SKMiniBatchKMeans`, `SKDBSCAN` and the later
  silhouette scorer; the convenient scorer form no longer fabricates assignments from
  rounded floats.
- **Acceptance criteria:** every scenario is a contract test in `sciencekit_common`; no
  numeric algorithm is introduced.
