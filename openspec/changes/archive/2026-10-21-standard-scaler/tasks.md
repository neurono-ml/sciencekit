# Tasks — standard-scaler

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Kernel — `Σx²` axis reduction

- [x] 1.1 Red: companion tests for `sk_axis_sum_squares` (axis 0 and axis 1 totals, f32/f64,
  parallel ≡ sequential). *(commit `feat(math): add sk_axis_sum_squares reduction kernel`)*
- [x] 1.2 Implement `sk_axis_sum_squares` with the sequential form and the race-free
  per-chunk parallel form, sharing `SK_AXIS_SUM_GRAIN`; export it from the crate root.
- [x] 1.3 Green: parallel Σx² matches the sequential reference within tolerance.

## 2. Estimator, model and dense path

- [x] 2.1 Red: companion tests for `SKStandardScaler` (fit learns per-column mean/scale,
  transform z-scores, `with_mean`/`with_std` masking, zero-variance column, f32/f64,
  empty-input and sparse-input errors, transform column mismatch, backend default).
  *(commit `feat(preprocessing): add standard scaler estimator, model and tests`)*
- [x] 2.2 `SKStandardScalerBuilder` (`with_mean`, `with_std`, `backend`, `execution_mode`;
  private constructor) and the `SKBuilder` impl.
- [x] 2.3 `SKStandardScaler` estimator and the `SKUnsupervisedFit` impl building a
  `SKStandardScalerModel` from learned moments.
- [x] 2.4 `SKStandardScalerModel` with `SKFeatureTransformer`, `derive_statistics`, the
  `finalize_column` masking helper, `sample_count`/`mean`/`scale`/`backend` accessors.
- [x] 2.5 Green: dense fit/transform behave per spec; the estimator stays generic over
  `F: SKFloat`.

## 3. Backend-selection capability

- [x] 3.1 Publish `openspec/specs/backend-selection/spec.md` (closed-enum builder
  parameter, per-operation dispatch, model carries the kind). *(commit
  `docs(spec): add backend-selection spec and builder rule`)*
- [x] 3.2 Record the builder rule in `AGENTS.md` and `openspec/config.yaml`.
- [x] 3.3 Green: `backend_override_reaches_model` — the configured kind reaches estimator
  and fitted model and every operation's span.

## 4. Out-of-core streaming fit

- [x] 4.1 Red: streaming companion tests (uneven batches ≡ dense, single batch, empty
  source error, column-mismatch error, builder flags, f32, large scale). *(commit
  `feat(preprocessing): add out-of-core streaming fit for standard scaler`)*
- [x] 4.2 `StreamingMoments` state and the exact `merge_moments`/`merge_batch` (Chan).
- [x] 4.3 The per-batch update closure (`update_moments`, `batch_moments`,
  `batch_parallelism`) driven by the Phase 0 streaming executor.
- [x] 4.4 `fit_streaming` plan resolution (unbounded context → streaming branch; in-memory
  intent → collect and reuse the dense path).
- [x] 4.5 Green: streaming ≡ dense within relative tolerance on small and large data.

## 5. Shared deterministic test datasets

- [x] 5.1 Extract `test_datasets` (small hand-checkable matrix, large 100 000 × 8 matrix
  with a constant column, `chunk_rows`, `VecSource`/`chunked_source`, `assert_close`).
  *(commit `test(preprocessing): share deterministic small and large datasets across
  scaler tests`)*
- [x] 5.2 Refactor the dense and streaming tests to consume the shared datasets/helpers.

## 6. Incremental `partial_fit`

- [x] 6.1 Red: incremental companion tests (after-fit ≡ dense, single-row accumulation,
  empty batch no-op, column mismatch, chaining, builder flags, large scale). *(commit
  `feat(preprocessing): add incremental partial_fit to standard scaler model`)*
- [x] 6.2 Split `model.rs` into the `model/` folder module (`mod.rs` + `partial_fit.rs`)
  and implement `partial_fit` on the model via the shared `merge_moments`.
- [x] 6.3 Green: incremental ≡ dense within tolerance; `sample_count` accumulates.

## 7. Reconciliation — explicit shape field names

- [x] 7.1 Rename `SKOperationAttributes.rows`/`columns` to `rows_count`/`columns_count`
  and update every constructor and test; span field names stay `rows`/`columns`.
  *(commit `refactor(common): name operation shape fields rows_count/columns_count`)*

## 8. Acceptance and gates

- [x] 8.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [x] 8.2 Small and large data validated; concurrency validated (the immutable estimator and
  the fitted model are shared across threads with identical results).
- [ ] 8.3 Model export (Safetensors/JSON) and metric generation deferred: no
  `sciencekit_interop` (Phase 7) or metrics-suite (W6.3) crate exists yet. The fitted model is
  already the distinct, backend-carrying export boundary this change establishes; tracked for
  the interop/metrics changes.

## 9. Independent-review fixes

- [x] 9.1 Fix the axis-1 parallel reduction bug in `sk_axis_sum`/`sk_axis_sum_squares`
  (iterated columns for the per-row path); add non-square axis-1 parallel regression tests.
  *(commit `fix(math): correct axis-1 parallel reduction to iterate rows`)*
- [x] 9.2 Make `model/mod.rs` a pure dispatcher (move implementation to
  `model/core_implementation.rs`) and remove the remaining manual index loops.
  *(commit `refactor(preprocessing): pure-dispatcher model module and review fixes`)*
- [x] 9.3 Make `from_moments` crate-internal, record the carried execution intent in
  `transform`, and add default-intent and concurrency tests.
