# Tasks — sgd-classifier-regressor

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Shared class weighting and the central error variant

- [x] 1.1 Red: companion tests for the relocated `SKClassWeight` (`None`, `Balanced`,
  explicit, wrong length, negative rejected, all-zero rejected) and for
  `SKError::UnsupportedOperation` (the message names the operation and the reason).
  *(commit `refactor(linear_model): share the class-weighting policy and add UnsupportedOperation`)*
- [x] 1.2 Move `SKClassWeight` and `sk_expand_class_weights` into
  `crates/sciencekit_linear_model/src/class_weight/`, re-export it from `logistic` so
  `logistic::SKClassWeight` keeps resolving, and add the `UnsupportedOperation` variant to
  `sciencekit_common`'s central enum.
- [x] 1.3 Green: the existing logistic suite passes unchanged and the new companion tests pass.

## 2. Stochastic-gradient substrate (`crates/sciencekit_linear_model/src/sgd/`)

- [x] 2.1 Red: companion tests for the loss catalogue — each classifier and regressor loss
  value and slope on hand-checkable points (margin outside the band, inside the band, zero
  residual, epsilon band), the probabilistic flag of the two probabilistic losses, and the
  family mismatch rejection. *(commit `feat(linear_model): add the stochastic-gradient loss catalogue`)*
- [x] 2.2 Implement the sealed `SKSGDLoss` trait, the classifier and regressor loss
  implementations and the two public enums.
- [x] 2.3 Red: companion tests for the penalty and configuration validation (negative
  `alpha`, out-of-range `l1_ratio`, non-positive tolerance / initial rate / power /
  validation fraction, zero iteration budget, `Optimal` with zero `alpha`, `l1_ratio`
  honored only by `ElasticNet`). *(commit `feat(linear_model): add the sgd configuration and proximal penalty`)*
- [x] 2.4 Implement `config.rs` (validation) and `penalty.rs` (the L2 gradient term and the
  L1 soft-threshold, intercept never penalized).
- [x] 2.5 Red: companion tests for the learning-rate schedules (constant, inverse scaling
  monotone decay, optimal asymptotic form, adaptive halving on a plateau, optimal rejects
  zero alpha, the counter continues across incremental calls). *(commit
  `feat(linear_model): add the sgd learning-rate schedules`)*
- [x] 2.6 Implement `learning_rate/`.
- [x] 2.7 Red: companion tests for the seeded shuffler (same seed same order, different seeds
  differ, empty/single row, index range covers every row exactly once, dense and CSR orders
  agree) and for the averaged iterate (running mean equals the explicit mean of visited
  iterates, averaging survives an incremental continuation). *(commit
  `feat(linear_model): add the sgd shuffler and averaged iterate`)*
- [x] 2.8 Implement `shuffle.rs` (`SKRandomGenerator` + Fisher–Yates) and `averaging.rs`.
- [x] 2.9 Red: companion tests for the single-sample update and the running objective
  (one update equals the closed-form step, L1 produces exact zeros, the intercept is not
  penalized, a zero slope is a no-op, the running loss equals the mean of batch losses).
  *(commit `feat(linear_model): add the sgd single-sample update and epoch objective`)*
- [x] 2.10 Implement `state.rs`, `update.rs` and `objective.rs`.
- [x] 2.11 Red: companion tests for the three epoch drivers (dense, CSR and batch agree
  under the same seed; CSR never densifies; the batch driver visits rows in order and stops
  on the final block). *(commit `feat(linear_model): add the sgd epoch drivers`)*
- [x] 2.12 Implement `epoch/dense.rs`, `epoch/sparse.rs`, `epoch/batch.rs` and the dispatcher.
- [x] 2.13 Red: companion tests for the convergence bookkeeping (a plateau converges before
  the budget, `Strict` budget exhaustion raises `NotConverged { iterations }`, `Permissive`
  returns the last iterate with `converged() == false` and a warning, a provider-limited fit
  never raises, early stopping keeps the best iterate, early stopping requires more than one
  epoch). *(commit
  `feat(linear_model): add the sgd convergence policy and early-stopping bookkeeping`)*
- [x] 2.14 Implement `early_stopping.rs` and the epoch bookkeeping, including the
  `SKConvergencePolicy { Strict, Permissive }` selection.
- [x] 2.15 Green: the full substrate suite passes on both floats.

## 3. `SKSGDClassifier`

- [x] 3.1 Red: companion tests for the builder (defaults, every setter, validation errors,
  configuration flows into the model) and for a binary dense fit (a separable design is
  learned exactly, decisions order the labels, `f32`/`f64`). *(commit
  `feat(linear_model): add SKSGDClassifier with a binary one-vs-rest fit`)*
- [x] 3.2 `SKSGDClassifierBuilder<F>` + the `SKBuilder` impl (private constructor,
  validation, warm-start source).
- [x] 3.3 `SKSGDClassifier<F>` + the `SKSupervisedFit<F>` impl (shape validation, target
  canonicalization through `SKLabelTable`, representation dispatch, `Fit` span, backend
  transfer).
- [x] 3.4 `SKSGDClassifierModel<F>` (accessors including `converged()` and `epochs_taken()`,
  `SKClassifierPredictor<F>`, `decision_function`, the class table, probabilities for the
  probabilistic losses and `UnsupportedOperation` otherwise).
- [x] 3.5 Green: binary fit/predict/probabilities behave per spec.
- [x] 3.6 Red: companion tests for multi-class one-vs-rest (three classes produce three
  heads, arg-max labels, probability rows sum to one for `log_loss`, single head for two
  classes). *(commit `feat(linear_model): add one-vs-rest multi-class fits to SKSGDClassifier`)*
- [x] 3.7 Implement the per-head fitting and prediction.
- [x] 3.8 Green: the multi-class suite passes.
- [x] 3.9 Red: companion tests for every classifier loss (separable exactness for `hinge` and
  `perceptron`, `squared_hinge` and `modified_huber` convergence, `log_loss` calibration,
  probabilities refused for non-probabilistic losses). *(commit
  `feat(linear_model): cover every classifier loss in SKSGDClassifier`)*
- [x] 3.10 Wire the loss catalogue through the fitting path and prediction.
- [x] 3.11 Red: companion tests for class weighting (`Balanced` improves minority recall,
  explicit weights in class order, negative rejected) and warm start (seeded refit starts
  from the source and never regresses within the same budget, misaligned source rejected).
  *(commit `feat(linear_model): add class weighting and warm start to SKSGDClassifier`)*
- [x] 3.12 Implement the weighting expansion and the warm-start seeding.
- [x] 3.13 Green: weighting and warm-start suites pass on both floats.

## 4. `SKSGDRegressor`

- [x] 4.1 Red: companion tests for the builder (defaults, setters, validation, configuration
  flows into the model) and for a dense squared-error fit (closed-form step, prediction
  matches the linear predictor, prediction shape error, integer targets promoted, nominal
  rejected, `f32`/`f64`). *(commit `feat(linear_model): add SKSGDRegressor with a squared-error fit`)*
- [x] 4.2 `SKSGDRegressorBuilder<F>` + the `SKBuilder` impl.
- [x] 4.3 `SKSGDRegressor<F>` + the `SKSupervisedFit<F>` impl.
- [x] 4.4 `SKSGDRegressorModel<F>` (accessors including `converged()` and `epochs_taken()`,
  and the `SKRegressorPredictor<F>` impl).
- [x] 4.5 Green: dense fit/predict behaves per spec.
- [x] 4.6 Red: companion tests for the robust and insensitive losses (large epsilon huber ≡
  squared error, insensitive band keeps the start, huber resists an outlier better than
  squared error, epsilon zero insensitive approaches absolute error). *(commit
  `feat(linear_model): cover the huber and insensitive losses in SKSGDRegressor`)*
- [x] 4.7 Wire the regressor loss catalogue through fitting.
- [ ] 4.8 Red: companion tests for convergence to OLS (`squared_error`, `alpha(0.0)`,
  generous budget, agreement with `SKLinearRegression`) and for the L1/L2/ElasticNet
  penalties on the regressor. *(commit `feat(linear_model): add the penalty portfolio and OLS convergence to SKSGDRegressor`)*
- [ ] 4.9 Green: the regressor suite passes on both floats.

## 5. Sparse input (both estimators)

- [ ] 5.1 Red: companion tests fitting the same design densely and as CSR with a fixed seed
  and shuffling disabled — coefficients, intercepts, decisions and probabilities agree across
  every loss and penalty, and no `n × p` dense copy is built. *(commit
  `feat(linear_model): add zero-copy CSR fits to the sgd estimators`)*
- [ ] 5.2 Wire `SKDataView::Sparse` through the estimators and the sparse epoch driver.
- [ ] 5.3 Green: sparse and dense agree within the documented relative tolerance.

## 6. Out-of-core and incremental fits (both estimators)

- [ ] 6.1 Red: companion tests for `fit_streaming` (one pass equals one in-memory epoch in
  the same order, uneven batches, single batch, empty source error, column mismatch, early
  stopping refused with `UnsupportedOperation`, both floats) and for retention (a source
  yielding many batches leaves only one batch plus the state alive). *(commit
  `feat(linear_model): add sgd fit_streaming over the supervised source`)*
- [ ] 6.2 Implement the streaming drivers for both estimators over the batch epoch driver.
- [ ] 6.3 Red: companion tests for `partial_fit` (continuation improves or preserves the
  objective, matches the equivalent concatenated fit within tolerance, the schedule counter
  continues, the average continues, a column mismatch is rejected, an unseen class is
  rejected, both floats). *(commit `feat(linear_model): add partial_fit continuation to the sgd models`)*
- [ ] 6.4 Implement `partial_fit` on both fitted models.
- [ ] 6.5 Green: streaming and incremental suites pass.

## 7. Shared deterministic test datasets

- [ ] 7.1 Extract the shared datasets and helpers (a separable classifier design, an
  imbalanced design, a multi-class design, a well-conditioned regression design with a known
  OLS solution, an outlier-corrupted design, dense and CSR forms, fixed seeds,
  `assert_close`). *(folded into the corresponding feature commits)*
- [ ] 7.2 Refactor the classifier, regressor, sparse, streaming and incremental suites onto
  the shared datasets.

## 8. Acceptance and gates

- [ ] 8.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [ ] 8.2 Small and large data validated; concurrency validated (the immutable estimator and
  model are shared across threads with identical results); dense, sparse, streaming and
  incremental fits cross-agree within the documented tolerance; probabilities are normalized
  and restricted to the probabilistic losses; regression converges to the OLS solution.
- [ ] 8.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted models
  are the distinct, backend-carrying export boundaries this change establishes; tracked for
  the interop/metrics changes.

## 9. Documentation, changelog and review

- [ ] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for `SKSGDClassifier`, `SKSGDRegressor`, the `stochastic-gradient-solver` substrate, the
  shared class weighting and the `UnsupportedOperation` error variant.
- [ ] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): the two estimator entries in `docs/src/algorithms.md` and a new
  `Development Explained` chapter (`docs/src/development-explained/`) registered in
  `SUMMARY.md`, walking through the per-sample update, the loss catalogue, the schedules,
  averaging, early stopping and the streaming regime — including the `maintaining.md`
  pointer.
- [ ] 9.3 Independent-review fixes: address every finding (spec conformance, numerics,
  convergence criteria, representation dispatch, naming, test coverage) and re-run the gates.
