## Why

Wave 2 built the *regression* half of `sciencekit_linear_model`: `SKLinearRegression`
(W2.1), `SKRidge` (W2.2) and `SKLasso`/`SKElasticNet` (W2.3). All three are regressors: they
share `SKSupervisedFit<F>`, predict a continuous response through `SKRegressorPredictor<F>`,
and reduce to a *linear* system at stationarity. W2.4 introduces `SKLogisticRegression`, the
library's **first classifier**, which breaks three assumptions those changes could rely on:

1. the logistic negative log-likelihood is **nonlinear** in the coefficients — no
   normal-equation, SVD or Cholesky solve applies;
2. a classifier's surface is **labels and probabilities** (`SKClassifierPredictor<F>`:
   `predict_labels -> Array1<i64>`, `predict_probabilities -> Array2<F>`), with integer or
   nominal targets canonicalized through `SKLabelTable`;
3. the model is **multi-headed** — one coefficient vector per class under multinomial /
   one-vs-rest, not a single `n_features` vector.

The wave plan (`wave-plan-foundation`, task 4) scopes W2.4 as *"`SKLogisticRegression`
(pure-Rust L-BFGS-B, SAGA for L1, multinomial)"*, anchored on Defazio et al. 2014 (SAGA) and
Fan et al. 2008 (LIBLINEAR). This change realizes the full scikit-learn parity surface
requested for the estimator: `penalty ∈ {l2, l1, elasticnet, none}`, `l1_ratio`, `C`/`alpha`,
`multi_class ∈ {auto, ovr, multinomial}`, `class_weight`, `warm_start`, `fit_intercept`, and
a solver portfolio `{lbfgs, liblinear, saga}`.

## What Changes

- **New `logistic_regression` module in `crates/sciencekit_linear_model`** holding
  `SKLogisticRegression<F>` (builder-only estimator) and the distinct fitted
  `SKLogisticRegressionModel<F>` (learned state and export boundary).
- **New `logistic` solver substrate** (its own module tree) shared by the estimator: the
  objective configuration and validation, the penalty/solver/multi-class resolution, the
  class-weight normalization, the pure-Rust **L-BFGS-B with an orthant-wise (OWL-QN) step**
  for the smooth `l2`/`none` path, the pure-Rust **SAGA** (variance-reduced incremental
  gradient) solver for `l1`/`elasticnet`/`none` and the sparse path, and a
  **LIBLINEAR-style dual coordinate-descent** solver for the `l1`/`l2` one-vs-rest path.
- **Builder surface:** `penalty(SKLogisticPenalty)` (default `L2`), `l1_ratio(F)` (default
  `0.5`, used only with `ElasticNet`), **both** `C(F)` (default `1.0`, inverse
  regularization) **and** `alpha(F)` (direct regularization, `alpha = 1/C`) — the last setter
  wins and a **warning** is emitted when both are used on the same builder — `solver(
  SKLogisticSolver)` (default `Lbfgs`), `multi_class(SKMultiClass)` (default `Auto`),
  `class_weight(SKClassWeight)` (default `None`), `warm_start(bool)` (default `false`),
  `include_intercept(bool)` (default `true`), `tolerance(F)` (default `1e-4`),
  `maximum_number_of_iterations(usize)` (default `100`), `backend(SKBackendKind)` (default
  `Faer`) and `execution_mode(SKExecutionMode)` (default `Automatic`). The direct constructor
  stays private.
- **Objective.** Binary fit minimizes
  `(1/(2·C))·R(w) + (1/n)·Σ_i c_i·log(1 + exp(−y_i·x_iᵀw))` with `y_i ∈ {−1, +1}`;
  multinomial minimizes the softmax cross-entropy with the same penalty; `R(w)` is
  `‖w‖₂²` (`L2`), `‖w‖₁` (`L1`), `l1_ratio·‖w‖₁ + (1−l1_ratio)/2·‖w‖₂²` (`ElasticNet`) or
  `0` (`None`). `c_i` is the per-sample class weight. The intercept is never penalized.
- **Multi-class resolution:** `auto` resolves to **multinomial** for the smooth solvers
  (`lbfgs`, `saga`) on more than two classes, and to **one-vs-rest** for `liblinear` and for
  binary problems; `ovr` and `multinomial` force the strategy. `liblinear` with `multinomial`
  is a structured error.
- **Targets and labels:** integer (`i64`) and nominal (`&str`) targets are canonicalized
  through `SKLabelTable`; the fitted model exposes the class table and maps canonical indices
  back to labels. Continuous targets are rejected.
- **Prediction surface:** the model implements `SKClassifierPredictor<F>` (`predict_labels`,
  `predict_probabilities`) and a `decision_function`; probabilities are the exact
  sigmoid (binary) / softmax (multinomial) / normalized OvR of the linear predictor.
- **Dense and CSR sparse input:** SAGA and LIBLINEAR consume zero-copy `CsMatView` without
  densifying; L-BFGS-B runs on dense and sparse designs through the shared design
  abstraction.
- **Incremental `partial_fit` and out-of-core `fit_streaming`:** because the logistic
  objective is not a function of the Gram matrix, both regimes **retain the design** (dense
  rows or stacked CSR batches) in a shared accumulator and re-solve at read time; the memory
  trade-off is documented and batches of any size or order converge to the dense fit.
- **`warm_start` as an explicit source:** `warm_start(true)` seeds the optimizer from
  coefficients supplied on the builder (a previous model's solution); the estimator stays
  immutable, stateless and `Send + Sync` — a documented deviation from scikit-learn's
  instance-state reuse.
- **Companion test modules** (`*_tests.rs`) covering binary/multinomial/OvR, every penalty ×
  solver combination, dense/sparse agreement, hand-checkable closed forms, probability
  normalization and calibration, label round-tripping, class weights, warm start, `C`/`alpha`
  precedence, incremental and streaming equivalence, `f32`/`f64`, concurrency and the model
  boundary.

## Capabilities

### New Capabilities

- `logistic-solver`: the shared `sciencekit_linear_model` substrate — the objective
  configuration and validation, the penalty/solver/multi-class resolution, the class-weight
  normalization, the L-BFGS-B/OWL-QN, SAGA and LIBLINEAR dual coordinate-descent solvers, the
  shared dense/sparse design abstraction and the retained-design accumulator.
- `logistic-regression`: the `SKLogisticRegression` estimator/model contract — builder surface
  and defaults, the `C`/`alpha` precedence and warning, the penalty semantics, the multi-class
  strategies, target canonicalization, prediction (labels/probabilities/decision function),
  `class_weight`, `warm_start`, the incremental/streaming regimes and the error taxonomy.

### Modified Capabilities

_None._ The change adds new capabilities; it does not change the requirements of
`linear-regression`, `ridge-regression`, `lasso`, `elastic-net`, `coordinate-descent-solver`,
`estimator-contracts`, `scoring-contracts`, `backend-selection`, `observability` or
`streaming-batches`. The classifier contracts (`SKClassifierPredictor`, `SKLabelScorer`) and
`SKLabelTable` already exist in `sciencekit_common`; this change consumes them rather than
extending them.

## Impact

- **New code:** `crates/sciencekit_linear_model/src/logistic_regression/` (`builder`,
  `core_implementation`, `model/`, `streaming/` + companion `*_tests.rs`) and
  `crates/sciencekit_linear_model/src/logistic/` (`config`, `solver/lbfgs`, `solver/saga`,
  `solver/liblinear`, `design`, `accumulator`, `class_weight` + companion `*_tests.rs`).
- **Dependencies:** none new — `ndarray`, `sprs` and `rayon` are already production
  dependencies of `sciencekit_linear_model`. The solvers are pure Rust (no LAPACK/BLAS
  requirement for the default path).
- **Public API:** new exports `SKLogisticRegression`, `SKLogisticRegressionBuilder`,
  `SKLogisticRegressionModel`, `SKLogisticPenalty`, `SKLogisticSolver`, `SKMultiClass`,
  `SKClassWeight` — all additive, no existing item changes.
- **Downstream:** the logistic solver substrate (variance-reduced incremental gradient) is
  the natural base for `SKSGDClassifier`/`SKSGDRegressor` (W2.5) and the linear SVM (W3);
  `docs/src/algorithms.md` gains the `SKLogisticRegression` entry and a new Developer Book
  chapter documents it.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data; dense,
  incremental and streaming fits agree within float tolerance; sparse and dense agree within
  a documented relative tolerance; probabilities are normalized and calibrated; safe under
  concurrency; the model carries its backend kind and is the export boundary. Model export
  (Safetensors/JSON) and the metrics suite are deferred to their Phase 7 / W6.3 changes, as
  the scaler, W2.1, W2.2 and W2.3 changes already do.
