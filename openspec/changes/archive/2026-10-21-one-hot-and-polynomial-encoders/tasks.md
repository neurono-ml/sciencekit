# Tasks

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — folder modules with a
pure-dispatcher `mod.rs`. One commit per task.

## 1. Categorical feature view in `sciencekit_common`

- [x] 1.1 Red: companion tests for `SKNominalMatrixView` and `SKCategoricalView` (row-major
  shape, `value(row, column)` borrows the label, `is_empty`, representation label, nominal vs
  integer dispatch, conversion via `TryInto`/`From` for `&[&[&str]]` and `ArrayView2<i64>`).
  Verify: tests fail before the types exist.
- [x] 1.2 Implement `SKNominalMatrixView<'a>` and the non-exhaustive `SKCategoricalView<'a>`
  (with `representation()` and once-per-operation rejection) plus the conversion impls, exported
  from the crate root. Verify: the new tests pass and `cargo test -p sciencekit_common` is green.
- [x] 1.3 Export the documented `SK_CATEGORICAL_MISSING_CODE: i64 = -1` constant and assert in a
  test that it is reachable from the crate root. Verify: the constant test passes.

## 2. OneHotEncoder — estimator, model and dense/sparse transform

- [x] 2.1 Promote `sprs` to the workspace pins and to `sciencekit_preprocessing` production
  dependencies. Verify: `cargo build -p sciencekit_preprocessing` succeeds.
- [x] 2.2 Red: companion tests for `SKOneHotEncoder` (hand-checkable single text column
  `["a","b","a"] → [[1,0],[0,1],[1,0]]`, output width = category sum, first-occurrence table
  order, deterministic tables across fits, table roundtrip, unknown-category `Reject` error,
  empty input, inconsistent column counts, backend default/override, f32/f64). Verify: tests
  fail.
- [x] 2.3 `SKOneHotEncoderBuilder` (`sparse_output`, `handle_unknown`, `backend`,
  `execution_mode`; private constructor) and the `SKBuilder` impl. Verify: default configuration
  is asserted by tests.
- [x] 2.4 `SKOneHotEncoder` estimator (dense `fit` over `SKCategoricalView`), the per-column
  `SKOneHotColumnCategories` (nominal `SKLabelTable` / integer code table) and
  `SKOneHotEncoderModel` with accessors and carried backend. Verify: the dense fit tests pass.
- [x] 2.5 Dense `transform_dense` + `transform()` returning `SKEncodedOutput<F>`, honoring the
  `handle_unknown` policy and the missing sentinel. Verify: dense output and policy tests pass.
- [x] 2.6 Parallel two-pass CSR `transform_sparse` (rayon row-nnz pass + fill,
  `CsMat::new_from_unsorted`). Verify: the sparse ≡ dense equivalence tests pass exactly.

## 3. OneHotEncoder — out-of-core fit_streaming

- [x] 3.1 Red: companion tests for `fit_streaming` (uneven categorical batches ≡ dense tables,
  first-occurrence order across the concatenation, inconsistent-columns error, empty sequence
  error). Verify: tests fail.
- [x] 3.2 Implement `fit_streaming` over an iterator of `SKCategoricalView` batches reusing the
  table-building logic. Verify: streaming ≡ dense tests pass exactly.

## 4. PolynomialFeatures — estimator, model and lazy expansion

- [x] 4.1 Red: companion tests for `SKPolynomialFeatures` (`[2,3]` with defaults → `[1,2,3,4,6,9]`,
  output width for two features degree two = 6, `interaction_only` drops pure powers,
  `include_bias(false)` drops the constant, documented column order, `degree(0)` →
  `InvalidHyperparameter`, empty/sparse-input errors, transform column mismatch, backend
  default/override, f32/f64, determinism). Verify: tests fail.
- [x] 4.2 `SKPolynomialFeaturesBuilder` (`degree`, `interaction_only`, `include_bias`,
  `sparse_output`, `backend`, `execution_mode`; private constructor, `degree` validation) and the
  `SKBuilder` impl. Verify: default configuration and invalid-degree tests pass.
- [x] 4.3 Exponent term generator (bias-first, increasing total degree, lexicographic within a
  degree, `interaction_only` restricting to `{0,1}`) with a test pinning the term list for the
  hand-checkable case. Verify: generator tests pass.
- [x] 4.4 `SKPolynomialFeatures` estimator and `SKPolynomialFeaturesModel` (learned input/output
  counts, term list, backend); dense `transform_dense` streaming term values row by row with
  `azip!`/`map`. Verify: dense expansion tests pass.
- [x] 4.5 Sparse `transform_sparse` (only nonzeros stored, parallel build) and `transform()`
  returning `SKExpandedOutput<F>` honoring `sparse_output`. Verify: sparse ≡ dense equivalence
  tests pass exactly.

## 5. PolynomialFeatures — out-of-core fit_streaming

- [x] 5.1 Red: companion tests for `fit_streaming` (uneven batches ≡ dense feature count,
  in-memory intent collects, empty source error). Verify: tests fail.
- [x] 5.2 Implement `fit_streaming` over `SKLazySource` (in-memory collect branch; otherwise learn
  the feature count from the first non-empty batch). Verify: streaming tests pass.

## 6. Documentation

- [x] 6.1 Add the doc-branch chapters: a new `Development Explained` chapter for the categorical
  boundary + both transformers (with Mermaid diagrams), register it in `SUMMARY.md` and update
  `maintaining.md`; document the public API. Verify: `mdbook build docs` passes with zero
  warnings.

## 7. Acceptance and gates

- [x] 7.1 `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` and
  `cargo test --workspace` all green; `mdbook build docs` zero warnings.
- [x] 7.2 Small and large data validated; concurrency validated (shared estimator/model across
  threads); dense and sparse outputs provably equal; models carry their backend kind and expose
  the label tables as the export boundary (export/metrics deferred to the interop/metrics changes
  as in `standard-scaler`).
- [x] 7.3 Independent review by a separate agent validating the spec was met; fix findings.

## Workflow follow-up

- Create the ADR issue and open the code, openspec and docs PRs referencing it; update
  `CHANGELOG.md` (English, keep-a-changelog 1.1.0) when the change lands; archive the change
  after the PRs merge.
