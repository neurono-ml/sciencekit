# Tasks — sparse-streaming-batches

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with
pure-dispatcher `mod.rs`. One commit per task; the mapping to history is noted per group.

## 1. Owned feature block (`sciencekit_common/src/batching/feature_block/`)

- [x] 1.1 Red: companion tests for `SKFeatureBlock` (dense and sparse construction, shape
  counts, `is_sparse`, zero-copy `view()`, byte size from the stored buffers, a sparse block
  moved across threads). *(commit `feat(common): add the owned streaming feature block`)*
- [x] 1.2 Implement `SKFeatureBlock<F>` over `Array2` and `CsMat`, with `view()`,
  `number_of_rows`, `number_of_columns`, `is_sparse` and `byte_size`.
- [x] 1.3 Red: companion tests for the representation-aware batches (`new_dense`/`new_sparse`,
  position and final flag preserved, supervised target-count mismatch rejected, dense
  constructors keep their meaning). *(commit `feat(common): carry feature blocks in streaming batches`)*
- [x] 1.4 Replace the `Array2` field of `SKDataBatch` and `SKSupervisedBatch` with
  `SKFeatureBlock`; expose `view()`, `number_of_rows`, `number_of_columns`, `is_sparse` and
  remove the dense-only accessors.
- [x] 1.5 Green: the block and batch suites pass on both floats, and the existing dense
  streaming tests pass unchanged.

## 2. Prefetch driver and memory accounting

- [x] 2.1 Red: companion tests for the driver (a sparse block reaches the update closure
  unchanged, block order is preserved, and the double-buffer guard uses the sparse byte
  size rather than `rows × columns × size_of`). *(commit
  `feat(common): pass feature blocks through the streaming driver`)*
- [x] 2.2 Update the prefetch driver and the plan resolution to consume and account for the
  block representation.
- [x] 2.3 Green: the streaming-executor suite passes, including the sparse memory-guard case.

## 3. Preprocessing estimators

- [x] 3.1 Red: conformance tests — streaming `SKStandardScaler`, `SKMinMaxScaler` and
  `SKRobustScaler` from sparse batches equals the dense stream and the dense `fit` on the
  concatenated data, with implicit zeros counted per column and no `n × p` dense copy.
  *(commit `feat(preprocessing): stream the scalers from sparse batches`)*
- [x] 3.2 Implement per-column CSR statistics for the three scalers.
- [x] 3.3 Red: conformance tests for `SKPolynomialFeatures` (sparse in, sparse out, dense
  agreement, no densification). *(commit
  `feat(preprocessing): stream polynomial features from sparse batches`)*
- [x] 3.4 Implement the sparse polynomial expansion.
- [x] 3.5 Green: the preprocessing conformance suite passes.

## 4. Imputation estimator

- [x] 4.1 Red: conformance tests for `SKSimpleImputer` over sparse batches (mean/median/mode/
  constant equal the dense stream; the `0.0` marker still claims implicit zeros; the missing
  indicator stacks after the imputed block, sparse or dense). *(commit
  `feat(impute): stream the simple imputer from sparse batches`)*
- [x] 4.2 Implement the sparse streaming path.
- [x] 4.3 Green: the impute conformance suite passes.

## 5. Linear models

- [x] 5.1 Red: conformance tests for `SKLinearRegression` and `SKRidge` over sparse batches
  (equals the dense stream and the dense `fit`; the Gram accumulation counts rows once; no
  densification). *(commit
  `feat(linear_model): stream OLS and ridge from sparse batches`)*
- [x] 5.2 Implement the sparse streaming path by reusing the in-memory sparse accumulators.
- [ ] 5.3 Red: conformance tests for `SKLasso`, `SKElasticNet`, `SKLogisticRegression`,
  `SKSGDClassifier` and `SKSGDRegressor` over sparse batches (retained design, class table,
  schedule counter and averaging preserved; no densification). *(commit
  `feat(linear_model): stream the remaining linear models from sparse batches`)*
- [ ] 5.4 Implement the sparse streaming paths.
- [ ] 5.5 Green: the linear-model conformance suite passes.

## 6. Categorical exception

- [ ] 6.1 Document `SKOneHotEncoder`'s categorical streaming seam as the deliberate
  exception in the book and the spec, and record an owned categorical block as a follow-up.

## 7. Acceptance and gates

- [ ] 7.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`,
  `cargo test --workspace` all green.
- [ ] 7.2 For every streaming estimator: sparse stream ≡ dense stream ≡ dense fit on the
  concatenated data; no dense `n × p` copy; the memory guard uses the sparse size; uneven
  batches converge to the same model; both floats and concurrency are covered.
- [ ] 7.3 Model export and metrics remain deferred to their Phase 7 / W6.3 changes.

## 8. Documentation, changelog and review

- [ ] 8.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0): **Changed** for the
  representation-aware batch accessors (breaking), **Added** for the owned feature block and
  sparse streaming.
- [ ] 8.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): a `Development Explained` chapter on the owned feature block and
  sparse out-of-core fits, registered in `SUMMARY.md`, plus the streaming chapter and
  `maintaining.md` updates; the PRD future-improvement note becomes a shipped feature.
- [ ] 8.3 Independent-review fixes: address every finding (statistics with implicit zeros,
  memory accounting, densification, migration completeness, naming, test coverage) and re-run
  the gates.
