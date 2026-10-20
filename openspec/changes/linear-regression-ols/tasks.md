# Tasks — linear-regression-ols

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Kernels — cross-product reductions

- [x] 1.1 Red: companion tests for `sk_gram_matrix` and `sk_gram_vector` (hand-checkable
  blocks, f32/f64, parallel ≡ sequential, symmetry of the Gram matrix).
  *(commit `feat(math): add sk_gram_matrix and sk_gram_vector cross-product kernels`)*
- [x] 1.2 Implement both with the sequential form and the race-free per-chunk parallel
  form, sharing the reduction grain; export them from the crate root.
- [x] 1.3 Green: parallel cross products match the sequential reference within tolerance.

## 2. Estimator, model and dense path

- [ ] 2.1 Red: companion tests for `SKLinearRegression` (exact line fit, intercept on/off,
  rank and singular values surfaced, rank-deficient minimum-norm solution, prediction,
  f32/f64, empty-input / sparse-input / target-length / prediction-shape errors, backend
  default). *(commit `feat(linear_model): add linear regression estimator, model and tests`)*
- [ ] 2.2 `SKLinearRegressionBuilder` (`include_intercept`, `backend`, `execution_mode`;
  private constructor) and the `SKBuilder` impl.
- [ ] 2.3 `SKLinearRegression` estimator and the `SKSupervisedFit<F>` impl building a
  `SKLinearRegressionModel<F>` through the backend's `lstsq` (design-matrix augmentation
  when `include_intercept`).
- [ ] 2.4 `SKLinearRegressionModel<F>` with `SKRegressorPredictor<F>` and the
  `coefficients`/`intercept`/`rank`/`singular_values`/`backend` accessors.
- [ ] 2.5 Green: dense fit/predict behave per spec; the estimator stays non-generic and the
  model stays generic over `F: SKFloat`.

## 3. Out-of-core streaming fit

- [ ] 3.1 Red: streaming companion tests (uneven batches ≡ dense, single batch, empty
  source error, column mismatch error, builder flags, f32, large scale).
  *(commit `feat(linear_model): add out-of-core streaming fit for linear regression`)*
- [ ] 3.2 `SKGramState` (Gram matrix, right-hand side, row count) and the per-batch
  accumulation through `sk_gram_matrix`/`sk_gram_vector`.
- [ ] 3.3 The per-batch update closure driven by the Phase 0 streaming executor.
- [ ] 3.4 `fit_streaming` plan resolution (unbounded context → streaming branch; in-memory
  intent → collect and reuse the dense path).
- [ ] 3.5 Green: streaming ≡ dense within relative tolerance on small and large data.

## 4. Incremental `partial_fit`

- [ ] 4.1 Red: incremental companion tests (after-fit ≡ dense, single-row accumulation,
  empty batch no-op, column mismatch, chaining, builder flags, large scale).
  *(commit `feat(linear_model): add incremental partial_fit to linear regression model`)*
- [ ] 4.2 Implement `partial_fit` on the model via the shared `SKGramState`, solving at
  read time.
- [ ] 4.3 Green: incremental ≡ dense within tolerance; the accumulated row count grows.

## 5. Shared deterministic test datasets

- [ ] 5.1 Extract `test_datasets` (small hand-checkable design with a known line, large
  synthetic design, `chunk_rows`, `VecSource`/`chunked_source`, `assert_close`).
  *(commit `test(linear_model): share deterministic small and large datasets across tests`)*
- [ ] 5.2 Refactor the dense, streaming and incremental tests to consume the shared
  datasets/helpers.

## 6. Acceptance and gates

- [ ] 6.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [ ] 6.2 Small and large data validated; concurrency validated (the immutable estimator and
  the fitted model are shared across threads with identical results).
- [ ] 6.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend-carrying export boundary this change establishes; tracked for
  the interop/metrics changes.

## 7. Independent-review fixes

- [ ] 7.1 Address every finding of the independent review (spec conformance, numerics,
  naming and test coverage) and re-run the gates.
