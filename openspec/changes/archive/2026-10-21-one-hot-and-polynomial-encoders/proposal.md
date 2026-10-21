# Proposal

## Why

Wave 1 (PRD §12 Phase 1) moves past the numeric scalers with the two feature-engineering
transformers scikit-learn users reach for next. `SKOneHotEncoder` turns categorical
features — text labels or small integer codes — into the numeric, model-ready indicator
representation every downstream estimator expects. `SKPolynomialFeatures` expands numeric
features into their polynomial and interaction terms, letting linear models fit nonlinear
structure without leaving the linear family. Both are everyday scikit-learn transformers, and
the wave plan (`wave-plan-foundation`, W1) pairs them in one change.

Two opportunities distinguish this change:

1. **Categorical features do not fit yesterday's boundary.** The data boundary
   (`data-view-boundary`) today covers only numeric dense/sparse features; categorical input
   was never representable. This change extends the boundary with a categorical feature view
   (nominal text columns and integer codes), so the encoder is a first-class, zero-copy
   consumer of the same seam every estimator uses — not a one-off side channel.
2. **Sparse-first output.** One-hot and polynomial expansions blow up dimensionality and are
   overwhelmingly sparse. The encoder and the polynomial expansion therefore emit a dense
   `Array2` **and** a `sprs` CSR matrix (parallel rayon build), so downstream algebra keeps
   the zero-copy sparse path the library already bets on.

## What Changes

- **New categorical feature view in `sciencekit_common`**: `SKNominalMatrixView<'a>` (a
  row-major matrix of borrowed text labels) and the non-exhaustive `SKCategoricalView<'a>`
  enum covering nominal text columns and integer-coded columns (`ArrayView2<'a, i64>`, with a
  documented missing-code sentinel). A universal `TryInto<SKCategoricalView<'a>>` seam mirrors
  the numeric boundary. This is a **modified** `data-view-boundary` capability.
- **New `SKOneHotEncoder`** in `sciencekit_preprocessing`, built only through
  `SKOneHotEncoder::build()`. It accepts **both** nominal text and integer-coded input through
  the categorical view, learns a deterministic per-column label table (first-occurrence order,
  reusing the `SKLabelTable` interning foundation), and emits a dense `Array2<F>` or a sparse
  `sprs` CSR matrix. Builder: `sparse_output(bool)` (default `true`, scikit-learn parity),
  `handle_unknown` policy (default reject), `backend(SKBackendKind)`, `execution_mode(...)`.
  Dense `fit`/`transform_dense`/`transform_sparse` plus out-of-core `fit_streaming` over an
  iterator of categorical batches.
- **New `SKPolynomialFeatures`** in `sciencekit_preprocessing`, built only through
  `SKPolynomialFeatures::build()`. Builder: `degree(usize)` (default 2),
  `interaction_only(bool)` (default `false`), `include_bias(bool)` (default `true`),
  `sparse_output(bool)` (default `false`), `backend(SKBackendKind)`, `execution_mode(...)`.
  Dense `fit`/`transform_dense`/`transform_sparse` that enumerate multi-index exponent terms
  with a lazy generator (no materialized combination blow-up), plus out-of-core `fit_streaming`
  over an `SKLazySource`.
- **Workspace dependency `sprs` promoted** from a dev-only dependency of
  `sciencekit_preprocessing` to a production dependency, and added to the workspace pins,
  so both transformers own the sparse output path.
- **Modified capability `data-view-boundary`**: adds the categorical/nominal feature view and
  its conversion seam, alongside the existing numeric dense/sparse view.
- Both estimators carry the configured `SKBackendKind` and record it in every operation span,
  per the `backend-selection` convention established by the `standard-scaler` change.

## Capabilities

### New Capabilities

- `one-hot-encoder`: `SKOneHotEncoder` behavior — categorical input acceptance (nominal text
  and integer codes) through the categorical view, builder surface and defaults, deterministic
  per-column label tables, dense and sparse (CSR) transform output equivalence, the
  `handle_unknown` policy, missing-code handling, out-of-core `fit_streaming` equivalence, and
  the error taxonomy.
- `polynomial-features`: `SKPolynomialFeatures` behavior — builder surface and defaults, the
  exponent multi-index expansion (`degree`, `interaction_only`, `include_bias`), the stable
  column ordering, dense and sparse (CSR) output equivalence, the streaming generator, the
  out-of-core `fit_streaming`, and the error taxonomy.

### Modified Capabilities

- `data-view-boundary`: adds the categorical feature view — a nominal (text) feature matrix
  view and integer-coded columns — as an extensible, non-exhaustive representation with a
  universal conversion seam and a precise unsupported-representation rejection path, mirroring
  the numeric feature view.

## Impact

- **Code:** `crates/sciencekit_common` gains the categorical feature view under `data_view/`;
  `crates/sciencekit_preprocessing` gains `one_hot_encoder/` and `polynomial_features/` folder
  modules (builder, core implementation, model, sparse build, streaming, companion `*_tests.rs`).
- **Dependencies:** `sprs` promoted to a production dependency of `sciencekit_preprocessing`
  and added to the workspace pins.
- **Downstream:** categorical pipelines (W2+), encoders feeding every numeric estimator, and the
  sparse output path reused by PCA/TruncatedSVD and the SVM/kernel work. The categorical view
  is reused by `SKKNNImputer` and future codecs.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data, dense and
  sparse outputs agree, streaming ≡ dense, safe under concurrency, and the fitted model carries
  its backend kind and exportable label tables as the export boundary.
