# Tasks — lasso-elasticnet-coordinate-descent

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Coordinate-descent substrate (`crates/sciencekit_linear_model/src/coordinate_descent/`)

- [ ] 1.1 Red: companion tests for the alpha-conversion trait, the sweep selection and the
  hyperparameter validation (scalar/sequence conversion, negative and empty alpha rejection,
  non-positive tolerance, zero iteration budget, `l1_ratio` bounds helper). *(commit
  `feat(linear_model): add the coordinate-descent configuration substrate`)*
- [ ] 1.2 Implement `SKIntoAlphas`, `SKSelection`, `SKCoordinateDescentConfig` and the shared
  `sk_validate_*` helpers.
- [ ] 1.3 Red: companion tests for the dense coordinate-descent solver (one-feature closed
  form, orthogonal closed form, KKT satisfaction, alpha monotonicity, zero alpha → least
  squares, large alpha → all zeros, `positive`, `selection(Random)` equivalence, `f32`/`f64`).
- [ ] 1.4 Implement the dense coordinate-descent solver: soft-threshold updates, the
  unpenalized residual-mean intercept coordinate, cyclic/random sweeps and convergence on the
  maximum coordinate change.
- [ ] 1.5 Red: companion tests for the sparse coordinate-descent solver (CSR equals dense
  within tolerance, both floats, a zero row is a no-op).
- [ ] 1.6 Implement the CSR column access and the shared column-access abstraction so dense
  and sparse share one coordinate loop without densifying the design.
- [ ] 1.7 Red: companion tests for gap-safe screening (screened equals unscreened, a positive
  screened count on a sparse solution, KKT reactivation recovers a deliberately screened-out
  coordinate, screening disabled equals enabled). *(commit
  `feat(linear_model): add gap-safe screening to the coordinate-descent solver`)*
- [ ] 1.8 Implement `screening.rs` (strong initial active set, dual-feasible gap radius,
  gap-safe elimination, KKT reactivation) and wire it into the solver.
- [ ] 1.9 Red: companion tests for the retained-design accumulator and the parallel per-alpha
  solve (dense and sparse accumulation equal concatenation, empty accumulator, column
  mismatch, parallel results equal sequential). *(commit
  `feat(linear_model): add the retained-design accumulator and parallel per-alpha solve`)*
- [ ] 1.10 Implement `SKCoordinateDescentState` (dense rows and stacked CSR batches) and the
  rayon per-alpha parallel solve.
- [ ] 1.11 Green: the full substrate suite passes on both floats, on small and large designs
  and across every branch (positive, selection, screening).

## 2. `SKLasso`

- [ ] 2.1 Red: companion tests for `SKLasso` (builder defaults, scalar/sequence alpha,
  intercept on/off, sparse fit, prediction, both floats, structured errors). *(commit
  `feat(linear_model): add SKLasso with dense and sparse coordinate descent`)*
- [ ] 2.2 `SKLassoBuilder<F>` + the `SKBuilder` impl (private constructor, hyperparameter
  validation).
- [ ] 2.3 `SKLasso<F>` + the `SKSupervisedFit<F>` impl (shape validation, representation
  dispatch, the `Fit` observability span, backend transfer).
- [ ] 2.4 `SKLassoModel<F>` (accessors, the `SKRegressorPredictor<F>` impl, single-alpha
  `predict` and `predict_with_alpha_index`).
- [ ] 2.5 Green: dense and sparse lasso fit/predict behave per spec.

## 3. `SKElasticNet`

- [ ] 3.1 Red: companion tests for `SKElasticNet` (builder defaults including `l1_ratio`,
  `l1_ratio = 1` equals lasso, the pure-L2 closed form at `l1_ratio = 0`, `l1_ratio`
  validation, sparse fit, both floats, structured errors). *(commit
  `feat(linear_model): add SKElasticNet with the l1_ratio mixing parameter`)*
- [ ] 3.2 `SKElasticNetBuilder<F>` + the `SKBuilder` impl with `l1_ratio` validated to
  `[0, 1]`.
- [ ] 3.3 `SKElasticNet<F>` + the `SKSupervisedFit<F>` impl.
- [ ] 3.4 `SKElasticNetModel<F>` (accessors including `l1_ratio`, the predictor).
- [ ] 3.5 Green: dense and sparse elastic-net fit/predict behave per spec.

## 4. Incremental and out-of-core fits

- [ ] 4.1 Red: streaming and incremental companion tests for both estimators (uneven batches
  equal dense, single batch, empty source error, column mismatch, in-memory intent collects,
  both floats, large scale). *(commit
  `feat(linear_model): add lasso and elastic-net fit_streaming and partial_fit`)*
- [ ] 4.2 Implement `fit_streaming` for both estimators over the retained-design accumulator.
- [ ] 4.3 Implement `partial_fit` on both fitted models.
- [ ] 4.4 Green: incremental and streaming fits equal the dense fit within tolerance.

## 5. Shared deterministic test datasets

- [ ] 5.1 Extract the shared coordinate-descent datasets and helpers (hand-checkable
  one-feature and orthogonal designs with known solutions, a large synthetic design, dense
  and CSR forms, `assert_close`). *(folded into the `feat(linear_model): add SKLasso ...`
  commit)*
- [ ] 5.2 Refactor the dense, sparse, streaming and incremental suites to consume the shared
  datasets and helpers.

## 6. Acceptance and gates

- [ ] 6.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [ ] 6.2 Small and large data validated; concurrency validated (the immutable estimators and
  the fitted models are shared across threads with identical results); dense, sparse,
  incremental and streaming fits cross-agree within the documented tolerance; the solution
  satisfies the KKT conditions.
- [ ] 6.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend-carrying export boundary this change establishes; tracked for
  the interop/metrics changes.

## 7. Documentation, changelog and review

- [ ] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for `SKLasso`, `SKElasticNet` and the `coordinate-descent-solver` substrate.
- [ ] 7.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): the `SKLasso`/`SKElasticNet` entries in `docs/src/algorithms.md` and
  a new `Development Explained` chapter (`docs/src/development-explained/`) registered in
  `SUMMARY.md`, walking through the coordinate-descent iteration, the unpenalized intercept,
  the gap-safe screening and the companion tests — including the `maintaining.md` pointer.
- [ ] 7.3 Independent-review fixes: address every finding (spec conformance, numerics,
  screening correctness, representation dispatch, naming, test coverage) and re-run the gates.
