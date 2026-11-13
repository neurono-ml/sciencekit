# Proposal

## Why

Wave 5's last item is anomaly detection. Isolation Forest is the standard non-parametric
outlier detector for tabular data: it isolates anomalies with random splits and needs no
distance metric or density estimate. It reuses the Wave 4 tree structure and completes the
unsupervised block.

## What Changes

- New sub-crate `sciencekit_outlier` with `SKIsolationForest<F>` and a distinct fitted
  model.
- Add an outlier prediction contract to `sciencekit_common`: `SKOutlierPredictor<F>` with
  `predict` returning a one-byte typed `SKOutlierLabel { Inlier, Outlier }`,
  `decision_function` and `score_samples`; integer `-1`/`1` exists only at the interop
  boundary.
- Builder-only surface: `number_of_trees` (default 100), `maximum_samples` (`Auto`
  default, meaning `min(256, n_samples)`), `contamination` (`Auto` default or a fraction
  in `(0, 0.5]`), `maximum_features` (default 1.0), `bootstrap` (default false),
  `random_seed` (default 0), `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`).
- Fit builds one isolation tree per subsample with random feature and random split
  threshold, computes path lengths, and derives the anomaly score
  `2^(-E[h(x)] / c(n))`; the offset is the `(1 − contamination)` quantile of the training
  scores.
- Dense and sparse inputs are accepted through the tree structure; sparse is never
  densified. Prediction, `fit_predict`, `decision_function` and `score_samples` are
  available.
- Validation: empty input, zero trees, invalid sample size, invalid contamination,
  non-finite values and feature-count mismatch.
- Explicit non-goals: Local Outlier Factor, One-Class SVM, missing values, streaming,
  `memmap2` and export (Wave 7).
- Companion tests: hand-checkable isolation depth, score monotonicity, contamination
  threshold, dense/sparse equivalence, determinism, concurrency, large data and metric
  production.

No existing estimator API changes beyond the additive outlier contract.

## Capabilities

### New Capabilities

- `outlier-prediction`: the model-side outlier contract — label convention, decision
  function and anomaly score.
- `isolation-forest`: `SKIsolationForest` — builder surface, random isolation trees,
  path-length scoring, contamination offset, representation parity, determinism and
  validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_outlier` workspace member; `sciencekit_common` gains the
  outlier predictor trait and re-export.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_tree`, `ndarray`,
  `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** completes Wave 5 and gives the pipeline an anomaly-detection stage.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
