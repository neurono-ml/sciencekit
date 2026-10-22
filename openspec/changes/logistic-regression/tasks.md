# Tasks — logistic-regression

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Logistic solver substrate (`crates/sciencekit_linear_model/src/logistic/`)

- [x] 1.1 Red: companion tests for the objective configuration, the hyperparameter validation
  (non-positive tolerance, zero iteration budget, non-positive regularization strength,
  out-of-range `l1_ratio`), the penalty/solver matrix resolution (every supported pair
  accepted, every unsupported pair rejected naming solver and penalty, `l1_ratio` honored only
  with `ElasticNet`) and the class-weight normalization (`None`, `Balanced`, explicit,
  negative rejected). *(commit `feat(linear_model): add the logistic configuration and resolution substrate`)*
- [x] 1.2 Implement `SKLogisticConfig`, the shared `sk_validate_*` helpers, the
  penalty/solver/multi-class resolution and the class-weight normalization.
- [x] 1.3 Red: companion tests for the objective (binary and softmax loss, gradient,
  proximal soft-threshold step) and the design abstraction (dense and CSR give the same
  dot/gradient, a CSR design is never densified, a zero row is a no-op).
- [x] 1.4 Implement `objective.rs` and `design.rs` (the dense/sparse trait with per-row dot,
  gradient accumulation and CSR row iteration).
- [x] 1.5 Red: companion tests for the L-BFGS-B/OWL-QN solver (a separable design's closed
  form, convergence on the projected-gradient norm, budget exhaustion raising
  `NotConverged { iterations }`, `L2` and `None`, both floats). *(commit
  `feat(linear_model): add the L-BFGS-B logistic solver`)*
- [x] 1.6 Implement `solver/lbfgs.rs` and `solver/convergence.rs`.
- [x] 1.7 Red: companion tests for the SAGA solver (L1 exact zeros on irrelevant features,
  ElasticNet, `None`, determinism under a fixed seed, sparse equals dense). *(commit
  `feat(linear_model): add the SAGA logistic solver`)*
- [x] 1.8 Implement `solver/saga.rs`.
- [x] 1.9 Red: companion tests for the LIBLINEAR dual coordinate-descent solver (agreement
  with L-BFGS under L2, shrinking active set, sparse equals dense, budget exhaustion).
  *(commit `feat(linear_model): add the LIBLINEAR dual coordinate-descent solver`)*
- [x] 1.10 Implement `solver/liblinear.rs`.
- [x] 1.11 Red: companion tests for the retained-design accumulator (dense rows and stacked
  CSR batches accumulate equal to concatenation, canonical labels and weights retained, empty
  accumulator errors). *(commit `feat(linear_model): add the logistic retained-design accumulator`)*
- [x] 1.12 Implement `accumulator.rs`.
- [x] 1.13 Green: the full substrate suite passes on both floats, on small and large designs
  and across every solver/penalty pair.

## 2. `SKLogisticRegression` — binary

- [x] 2.1 Red: companion tests for `SKLogisticRegression` (builder defaults, `C`/`alpha`
  precedence and the both-set warning, binary fit and prediction, integer and nominal targets
  canonicalized, structured errors). *(commit
  `feat(linear_model): add SKLogisticRegression with the binary fit`)*
- [x] 2.2 `SKLogisticRegressionBuilder<F>` + the `SKBuilder` impl (private constructor,
  hyperparameter validation, `C`/`alpha` last-setter precedence with the warning).
- [x] 2.3 `SKLogisticRegression<F>` + the `SKSupervisedFit<F>` impl (shape validation, target
  canonicalization through `SKLabelTable`, representation dispatch, the `Fit` observability
  span, backend transfer).
- [x] 2.4 `SKLogisticRegressionModel<F>` (accessors, the `SKClassifierPredictor<F>` impl,
  `decision_function`, the class table).
- [x] 2.5 Green: binary dense fit/predict behaves per spec, probabilities sum to one.

## 3. Multi-class strategies

- [x] 3.1 Red: companion tests for one-vs-rest and multinomial fits, `auto` resolution
  (multinomial for `Lbfgs`/`Saga` on three classes, one-vs-rest for `Liblinear` and binary),
  `Liblinear` + `Multinomial` rejected, probability normalization and ordering versus the
  decision function. *(commit `feat(linear_model): add one-vs-rest and multinomial logistic fits`)*
- [x] 3.2 Implement the one-vs-rest and multinomial strategies in the estimator/model.
- [x] 3.3 Green: the multi-class suite passes for three or more classes across the solvers.

## 4. Class weighting and warm start

- [x] 4.1 Red: companion tests for `class_weight` (`Balanced` balances the classes, explicit
  weights applied in class order, negative rejected) and `warm_start` (seeded refit equals the
  cold fit, `warm_start(true)` without a source equals the cold fit). *(commit
  `feat(linear_model): add class weighting and explicit warm start to SKLogisticRegression`)*
- [x] 4.2 Implement the per-sample weight expansion and the explicit warm-start seeding.
- [x] 4.3 Green: the weighting and warm-start suites pass on both floats.

## 5. Sparse input

- [x] 5.1 Red: companion tests fitting the same design densely and as CSR across every
  solver/penalty pair (coefficients, intercepts and probabilities agree; no `n × p` dense copy
  is built). *(commit `feat(linear_model): add zero-copy CSR fits to SKLogisticRegression`)*
- [x] 5.2 Wire the sparse representation through the estimator and the design abstraction.
- [x] 5.3 Green: sparse and dense agree within the documented relative tolerance.

## 6. Incremental and out-of-core fits

- [x] 6.1 Red: streaming and incremental companion tests (uneven batches equal dense, single
  batch, empty source error, column mismatch, in-memory intent collects, both floats, large
  scale). *(commit `feat(linear_model): add SKLogisticRegression fit_streaming and partial_fit`)*
- [x] 6.2 Implement `fit_streaming` over the retained-design accumulator.
- [x] 6.3 Implement `partial_fit` on the fitted model.
- [x] 6.4 Green: incremental and streaming fits equal the dense fit within tolerance.

## 7. Shared deterministic test datasets

- [ ] 7.1 Extract the shared logistic datasets and helpers (hand-checkable separable and
  orthogonal designs with known optima, an imbalanced design, a large synthetic design, dense
  and CSR forms, a fixed seed, `assert_close`). *(folded into the `feat(linear_model): add
  SKLogisticRegression ...` commit)*
- [ ] 7.2 Refactor the binary, multi-class, sparse, streaming and incremental suites to
  consume the shared datasets and helpers.

## 8. Acceptance and gates

- [x] 8.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [x] 8.2 Small and large data validated; concurrency validated (the immutable estimators and
  the fitted models are shared across threads with identical results); dense, sparse,
  incremental and streaming fits cross-agree within the documented tolerance; probabilities
  are normalized; the solution satisfies the logistic optimality (gradient) conditions.
- [x] 8.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend-carrying export boundary this change establishes; tracked for
  the interop/metrics changes.

## 9. Documentation, changelog and review

- [x] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for `SKLogisticRegression` and the `logistic-solver` substrate.
- [x] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): the `SKLogisticRegression` entry in `docs/src/algorithms.md` and a
  new `Development Explained` chapter (`docs/src/development-explained/`) registered in
  `SUMMARY.md`, walking through the logistic objective, the solver portfolio, the multi-class
  strategies, target canonicalization and the companion tests — including the
  `maintaining.md` pointer.
- [ ] 9.3 Independent-review fixes: address every finding (spec conformance, numerics, solver
  correctness, representation dispatch, naming, test coverage) and re-run the gates.
