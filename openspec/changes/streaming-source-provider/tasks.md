# Tasks — streaming-source-provider

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Provider contract (`sciencekit_common/src/batching/provider/`)

- [x] 1.1 Red: companion tests for the feature-only and supervised providers (a concrete
  source reports itself as not reopenable and yields once; a reopenable factory is invoked
  once per pass; a factory error propagates; both families agree on passes and order).
  *(commit `feat(common): add the streaming source provider contract`)*
- [x] 1.2 Implement `SKSourceProvider<F>` and `SKSupervisedSourceProvider<F>` (GAT source,
  `IS_REOPENABLE`, fallible `open`).
- [x] 1.3 Implement the residual `&mut S` provider for both families and
  `SKReopenableSource<G>` for both, exporting them from the crate root.
- [x] 1.4 Green: the provider suite passes on both floats.

## 2. Streaming executor over providers

- [x] 2.1 Red: companion tests for the provider entry points (one-shot provider runs one
  pass; reopenable provider runs the budget's passes; a budget above one over a one-shot
  provider runs one pass and records it; a factory failure aborts; block order is preserved
  inside a pass). *(commit `feat(common): drive streaming fits over source providers`)*
- [x] 2.2 Implement the provider-based streaming entry points, keeping the one-pass entry
  points as adapters for un-migrated estimators.
- [x] 2.3 Green: the executor suite passes and the existing streaming-batch tests are
  unchanged.

## 3. Preprocessing estimators

- [x] 3.1 Red: companion tests for `SKStandardScaler`, `SKMinMaxScaler` and
  `SKRobustScaler` streaming from a reopenable provider (the model equals the dense fit on
  the concatenated data; exactly one pass is taken and the statistics are not double
  counted). *(commit `refactor(preprocessing): stream the scalers over source providers`)*
- [x] 3.2 Migrate the three scalers' `fit_streaming` to the provider seam.
- [x] 3.3 Red: companion tests for `SKPolynomialFeatures` streaming from a reopenable
  provider (equals the dense fit, one pass). *(commit
  `refactor(preprocessing): stream polynomial features over source providers`)*
- [x] 3.4 Migrate `SKPolynomialFeatures`.
- [x] 3.5 Green: the preprocessing suites pass and the existing one-shot call shape compiles
  unchanged.

## 4. Imputation estimators

- [x] 4.1 Red: companion tests for `SKSimpleImputer` streaming from a reopenable provider
  (equals the dense fit, one pass, sketch-based median unchanged). *(commit
  `refactor(impute): stream the simple imputer over source providers`)*
- [x] 4.2 Migrate `SKSimpleImputer` and, if it exposes a streaming fit, `SKMissingIndicator`.
- [x] 4.3 Green: the impute suite passes.

## 5. Accumulating linear models

- [x] 5.1 Red: companion tests for `SKLinearRegression` and `SKRidge` streaming from a
  reopenable provider (equals the dense fit; one pass; the Gram state is not doubled).
  *(commit `refactor(linear_model): stream OLS and ridge over source providers`)*
- [x] 5.2 Migrate both.
- [x] 5.3 Red: companion tests for `SKLasso`, `SKElasticNet` and `SKLogisticRegression`
  streaming from a reopenable provider (equals the dense fit; one pass; the retained design
  is not doubled). *(commit
  `refactor(linear_model): stream the retained-design models over source providers`)*
- [x] 5.4 Migrate the three retained-design models.
- [x] 5.5 Green: the linear-model suites pass.

## 6. Stochastic estimators: multi-epoch streaming

- [x] 6.1 Red: companion tests for `SKSGDClassifier` and `SKSGDRegressor` streaming from a
  reopenable provider (the streamed fit with the same budget equals the in-memory fit within
  tolerance; a one-shot provider still takes one pass; `epochs_taken()` and `converged()`
  report the truth). *(commit
  `feat(linear_model): run multi-epoch sgd fits over reopenable sources`)*
- [x] 6.2 Migrate both SGD streaming drivers and remove the one-pass limitation from their
  specs (amend `sgd-classifier-regressor` in place if still open; otherwise add the delta
  here once it archives).
- [x] 6.3 Green: streamed and in-memory SGD fits agree under a fixed seed.

## 7. Categorical streaming seam

- [ ] 7.1 Document `SKOneHotEncoder`'s iterator seam as the deliberate exception and record
  the categorical provider contract as a follow-up in the book and the PRD's future list.
- [ ] 7.2 If the categorical contract proves trivial during review, migrate it; otherwise
  leave the documented exception.

## 8. Acceptance and gates

- [ ] 8.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [ ] 8.2 Every migrated estimator's streamed fit equals its dense fit on the concatenated
  data; a one-shot source is consumed exactly once; a non-iterative estimator's statistics
  are not double counted; a reopenable source over the same data reproduces the in-memory
  result; concurrency and both floats are covered.
- [ ] 8.3 Model export and metrics remain deferred to their Phase 7 / W6.3 changes, as the
  estimator changes already record.

## 9. Documentation, changelog and review

- [ ] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under
  `Unreleased → Added/Changed` for the provider seam, the reopenable source and the
  multi-epoch streaming regime.
- [ ] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): a `Development Explained` chapter on source providers, reopenable
  sources and the epoch rule, registered in `SUMMARY.md`, plus the streaming chapter and
  `maintaining.md` updates.
- [ ] 9.3 Independent-review fixes: address every finding (provider semantics, coherence,
  pass counts, migration completeness, naming, test coverage) and re-run the gates.
