# Tasks — ridge-regression

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Iterative-solver substrate (`sciencekit_math`)

- [x] 1.1 Red: companion tests for `SKLinearOperator` and `sk_conjugate_gradient` (a
  hand-checkable SPD system, agreement with a direct solve, parallel ≡ sequential,
  non-convergence on a zero budget, zero-right-hand-side). *(commit
  `feat(math): add SKLinearOperator and sk_conjugate_gradient`)*
- [x] 1.2 Implement the `SKLinearOperator<F>` trait (dimension + apply-into write) with a
  test-only dense operator used by the suite.
- [x] 1.3 Implement `sk_conjugate_gradient` (residual-based stopping, sequential and
  race-free parallel apply, `SKError::NotConverged { iterations }` on exhaustion).
- [x] 1.4 Red: companion tests for the zero-copy CSR normal-equation operator (product equals
  the explicitly formed `AᵀA` product, borrowed view is not copied, both floats).
- [x] 1.5 Implement the CSR normal-equation operator over a borrowed `CsMatView`; export the
  `solvers` items from the crate root.
- [x] 1.6 Green: the CG solve reaches tolerance on small and large SPD systems.

## 2. Ridge builder, estimator and model (dense)

- [x] 2.1 Red: companion tests for `SKRidge` (closed-form ridge solution on a hand-checkable
  system, scalar and array `alpha`, intercept on/off, unpenalized intercept, rank-deficient
  `Svd`, prediction, `f32`/`f64`, empty-input / target-length / negative-alpha / empty-alphas
  / prediction-shape errors, solver defaults). *(commit
  `feat(linear_model): add SKRidge with dense Cholesky and SVD solvers`)*
- [x] 2.2 `SKRidgeBuilder<F>` (`alpha`, `include_intercept`, `solver`, `tolerance`,
  `maximum_number_of_iterations`, `backend`, `execution_mode`; private constructor, hyperparameter
  validation) and the `SKBuilder` impl.
- [x] 2.3 The `SKRidge<F>` estimator and its `SKSupervisedFit<F>` impl: shape validation,
  representation dispatch, `Fit` observability span, solver resolution, transfer of the
  backend kind into the model.
- [x] 2.4 The centered-Gram construction (`XᵀX`, `Xᵀy`, `Σx`, `Σy`, row count) and the dense
  `Cholesky` solve `(Ĉ + αI)β = ĉ` for every alpha, with intercept recovery.
- [x] 2.5 The `Svd` solver (single centered decomposition, per-alpha application) and the
  `Automatic` resolution (per representation; Cholesky→Svd fallback on `alpha = 0` or factor
  failure).
- [x] 2.6 `SKRidgeModel<F>` with `alphas`/`coefficients`/`intercepts`/`solver`/`backend`
  accessors, `predict_with_alpha_index` and the `SKRegressorPredictor<F>` impl (single-alpha;
  structured error when several alphas are configured).
- [x] 2.7 Green: dense fit/predict behave per spec; the estimator is generic over `F` and the
  model carries the solver and backend.

## 3. Sparse CG solver

- [x] 3.1 Red: companion tests (CSR fit ≡ dense Cholesky within tolerance, `SparseCg` under
  `Automatic` for sparse input, never-densify guard, non-convergence error, both floats).
  *(commit `feat(linear_model): add the zero-copy sparse_cg ridge solver`)*
- [x] 3.2 The `SparseCg` solver: the augmented normal-equation operator
  (`DᵀD + αP`, `P = diag(0, 1, …, 1)`) built on the `sciencekit_math` substrate, per-alpha CG,
  tolerance/iteration plumbing and the non-convergence path.
- [x] 3.3 The `Svd`-rejects-sparse error path and the representation dispatch, with the solver
  exposed by the fitted model.
- [x] 3.4 Green: sparse ≡ dense within the documented relative tolerance.

## 4. Incremental and out-of-core fits

- [x] 4.1 Add the ridge centered accumulator `SKCenteredGramState` (`XᵀX`, `Xᵀy`, `Σx`,
  `Σy`, row count) beside W2.1's augmented-Gram state, which stays unchanged; the OLS tests
  stay green. *(commit `feat(linear_model): add the centered Gram accumulator for ridge`)*
- [x] 4.2 Red: streaming companion tests (uneven batches ≡ dense, single batch, empty source
  error, column mismatch, `SparseCg` intent falls back to the Gram solve, both floats, large
  scale). *(commit `feat(linear_model): add ridge fit_streaming and partial_fit`)*
- [x] 4.3 `fit_streaming` plan resolution and the accumulated centered Gram solve.
- [x] 4.4 `partial_fit` on the model through the shared accumulator, solving at read time.
- [x] 4.5 Green: incremental ≡ dense and streaming ≡ dense within tolerance.

## 5. Shared deterministic test datasets

- [x] 5.1 Extract shared ridge datasets/helpers (small hand-checkable system with a known
  solution, large synthetic design, dense and CSR forms, `assert_close`). *(folded into the
  `feat(linear_model): add SKRidge ...` commit)*
- [x] 5.2 Refactor the dense, sparse, streaming and incremental tests to consume the shared
  datasets/helpers.

## 6. Acceptance and gates

- [x] 6.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [x] 6.2 Small and large data validated; concurrency validated (the immutable estimator and
  the fitted model are shared across threads with identical results); the three solvers
  cross-agree within the documented tolerance.
- [ ] 6.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend- and solver-carrying export boundary this change establishes;
  tracked for the interop/metrics changes.

## 7. Documentation, changelog and review

- [x] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for `SKRidge` and the `iterative-linear-solvers` substrate.
- [x] 7.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): the `SKRidge` entry in `docs/src/algorithms.md` and a new
  `Development Explained` chapter (`docs/src/development-explained/`) registered in
  `SUMMARY.md`, walking through the solver portfolio, the unpenalized intercept and the
  companion tests — including the `maintaining.md` pointer.
- [x] 7.3 Independent-review fixes: address every finding (spec conformance, numerics, solver
  resolution, naming, test coverage) and re-run the gates.
