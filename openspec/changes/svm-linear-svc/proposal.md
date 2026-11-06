# Proposal

## Why

The SVM block closes with the linear maximum-margin baseline. `SKLinearSVC` is the
LIBLINEAR-style counterpart of `SKSVC(kernel = linear)` and the natural user of the
zero-copy CSR path the linear models already exercise. It is also the reference model for
large sparse text classification, where the kernel SVM's `n × n` cache is not viable.

## What Changes

- `SKLinearSVC<F>` in `sciencekit_svm`, sharing the crate with `SKSVC`/`SKSVR` (W3.5).
- Dual coordinate descent for the L2-regularized L1-loss and L2-loss (squared-hinge)
  linear SVC, following Fan et al. 2008 (LIBLINEAR), with one-vs-rest multiclass
  classifiers.
- Builder-only surface: `C` (default 1.0), `loss` (`SquaredHinge` default, `Hinge`),
  `class_weight` (`None` default, `Balanced`, explicit), `include_intercept` (default
  true), `tolerance`, `maximum_number_of_iterations`, `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`).
- Zero-copy sparse design: fit accepts dense or borrowed CSR, scans stored entries to
  form the dual coordinate updates and never densifies the matrix.
- Fitted model: one coefficient vector and one intercept per one-vs-rest head, the class
  table, the carried backend and convergence state. `decision_function` and
  `predict_labels` are exposed; `predict_probabilities` returns `UnsupportedOperation`
  because the linear SVC has no calibrated probability model.
- Validation: empty input, single-class targets, non-finite values, feature-count
  mismatch, non-positive `C`, zero tolerance/iteration budget and unsupported loss.
- Explicit non-goals: kernel functions, probability calibration, `memmap2` spilling and
  model export (Wave 7), and the `Crammer-Singer` multiclass formulation.
- Companion tests: hand-checkable binary and multiclass fixtures, dense/sparse
  equivalence without densification, class weights, determinism, convergence and metrics.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `linear-support-vector-classifier`: `SKLinearSVC` — builder surface, dual
  coordinate-descent solver contract, one-vs-rest multiclass behavior, class weights,
  decision function, representation parity, convergence and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_svm` gains the `linear_svc` modules; no new crate.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  `sciencekit_metrics` as a dev-dependency.
- **Downstream:** completes the Wave 3 SVM family and gives model selection a fast linear
  classifier for sparse text data.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Model export is
  deferred to Wave 7 and recorded in the PRD.
