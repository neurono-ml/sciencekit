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
- [x] 1.4 Fix the least-squares kernel to use the **thin** SVD (discovered: `SKFaerBackend::lstsq`
  used the full SVD, whose `U` is `m × m` — a 20 000-row design took ~12 s in debug); pin it
  with a tall-design `lstsq` regression test in `backend_tests.rs`.
  *(commit `fix(math): use the thin SVD in the least-squares kernel`)*
- [x] 1.5 Workspace `[profile.dev.package."*"] opt-level = 2` so dependency linear algebra
  stays fast in dev/test builds while our crates stay unoptimized.

## 2. Estimator, model and dense path

- [x] 2.1 Red: companion tests for `SKLinearRegression` (exact line fit, intercept on/off,
  rank and singular values surfaced, rank-deficient minimum-norm solution, prediction,
  f32/f64, empty-input / sparse-input / target-length / prediction-shape errors, backend
  default). *(commit `feat(linear_model): add SKLinearRegression with dense, streaming and
  incremental fits`)*
- [x] 2.2 `SKLinearRegressionBuilder` (`include_intercept`, `backend`, `execution_mode`;
  private constructor) and the `SKBuilder` impl.
- [x] 2.3 `SKLinearRegression` estimator and the `SKSupervisedFit<F>` impl building a
  `SKLinearRegressionModel<F>` through the backend's `lstsq` (design-matrix augmentation
  when `include_intercept`).
- [x] 2.4 `SKLinearRegressionModel<F>` with `SKRegressorPredictor<F>` and the
  `coefficients`/`intercept`/`rank`/`singular_values`/`backend` accessors.
- [x] 2.5 Green: dense fit/predict behave per spec; the estimator stays non-generic and the
  model stays generic over `F: SKFloat`.

## 3. Out-of-core streaming fit

- [x] 3.0 Add the supervised streaming contract to `sciencekit_common`
  (`SKSupervisedBatch`, `SKSupervisedLazySource`, `sk_run_supervised_streaming_driver`)
  and generalize the prefetch driver to serve both source kinds; spec delta on
  `streaming-batches`. *(commit `feat(common): add supervised streaming batches and prefetch driver`)*
- [x] 3.1 Red: streaming companion tests (uneven batches ≡ dense, single batch, empty
  source error, column mismatch error, builder flags, f32, large scale).
  *(folded into `feat(linear_model): add SKLinearRegression with dense, streaming and
  incremental fits`)*
- [x] 3.2 `SKGramState` (Gram matrix, right-hand side, row count) and the per-batch
  accumulation through `sk_gram_matrix`/`sk_gram_vector`.
- [x] 3.3 The per-batch update closure driven by the Phase 0 streaming executor.
- [x] 3.4 `fit_streaming` plan resolution (unbounded context → streaming branch; in-memory
  intent → collect and reuse the dense path).
- [x] 3.5 Green: streaming ≡ dense within relative tolerance on small and large data.

## 4. Incremental `partial_fit`

- [x] 4.1 Red: incremental companion tests (after-fit ≡ dense, single-row accumulation,
  empty batch no-op, column mismatch, chaining, builder flags, large scale).
  *(folded into `feat(linear_model): add SKLinearRegression with dense, streaming and
  incremental fits`)*
- [x] 4.2 Implement `partial_fit` on the model via the shared `SKGramState`, solving at
  read time.
- [x] 4.3 Green: incremental ≡ dense within tolerance; the accumulated row count grows.

## 5. Shared deterministic test datasets

- [x] 5.1 Extract `test_datasets` (small hand-checkable design with a known line, large
  synthetic design, `chunk_rows`, `VecSource`/`chunked_source`, `assert_close`).
  *(folded into `feat(linear_model): add SKLinearRegression with dense, streaming and
  incremental fits`)*
- [x] 5.2 Refactor the dense, streaming and incremental tests to consume the shared
  datasets/helpers.

## 6. Acceptance and gates

- [x] 6.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [x] 6.2 Small and large data validated; concurrency validated (the immutable estimator and
  the fitted model are shared across threads with identical results).
- [ ] 6.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend-carrying export boundary this change establishes; tracked for
  the interop/metrics changes.

## 7. Independent-review fixes

- [x] 7.1 Address every finding of the independent review (spec conformance, numerics,
  naming and test coverage) and re-run the gates.
  - **M1 (blocking):** the crate did not compile with `sciencekit_math/blas-backend` enabled;
    added `SKBackendSelection` / `sk_math_backend_for_kind` in `sciencekit_math` so the
    feature-gated dispatch stays inside the crate that owns the feature.
  - **m2/m3:** documented the regime semantics of `rank`/`singular_values` and the
    build-config `NdArrayLinalg` fallback.
  - **m4:** split `streaming_driver.rs` into the `streaming_driver/` folder module.
  - **m5:** added the missing tests (failing supervised source, block-survives-its-source,
    the minimum-norm property, the automatic default intent).
  - **m6/m7:** CHANGELOG entry and the `maintaining.md` pointer to the new chapter.
  - **n1/n5:** renamed `rhs` to `right_hand_side` and removed stray scratch files.
