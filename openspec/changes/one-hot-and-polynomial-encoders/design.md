# Design

## Context

See `proposal.md` — *Why*. The design is shaped by three existing facts of the codebase:

- The numeric data boundary (`sciencekit_common::data_view`) is `SKDataView<'a, F>` = dense
  (`ArrayView2<F>`) or sparse (`CsMatView<F>`), generic over `F: SKFloat`, entered through a
  `TryInto` seam. Categorical data has no representation there today; the only categorical
  vocabulary is `SKTargetView::Nominal(&[&str])` (1-D targets) and the `SKLabelTable`
  interning foundation in `label_table`.
- Every estimator follows the `standard-scaler` template: a non-generic configured estimator,
  a builder-only surface exposing `backend`/`execution_mode`, a distinct fitted model generic
  over `F`, and `*_tests.rs` companion modules under a folder module.
- `sprs` is already a dev-dependency of `sciencekit_preprocessing` (used in tests) but not a
  production dependency; the sparse output path needs it in production.

## Goals / Non-Goals

**Goals:**

- Add a categorical feature view to the shared boundary that is the categorical analogue of
  `SKDataView`: zero-copy, non-exhaustive, extensible, with the same `TryInto` seam and the
  same precise unsupported-representation rejection.
- Implement both transformers with a single, uniform sparse path (parallel CSR build) so
  dense and sparse outputs are provably equal and the sparse build is reusable by later
  sparse producers.
- Keep both estimators non-generic over scalar and backend, with `F` only on the fitted model
  and operation methods, per `backend-selection`.

**Non-Goals:**

- No `partial_fit` for `SKOneHotEncoder`: scikit-learn offers none (categories are unbounded),
  so `fit` + `fit_streaming` cover the incremental need by re-deriving tables; a per-batch
  merge would be semantically different and is deferred.
- No category-frequency pruning (`min_frequency`/`max_categories`) or `drop`
  (`first`/`if_binary`) in this change — deferred to keep scope minimal; the table structure
  leaves room for them.
- No string feature support in `SKPolynomialFeatures` (numeric only) and no sparse *input* for
  either transformer.
- No ONNX/Safetensors export implementation here; the model exposes the tables as the export
  boundary, and the interop layer (W7) consumes it.

## Decisions

### Decision 1 — Categorical view lives in `data_view`, as a sibling of `SKDataView`

Add to `sciencekit_common::data_view`:

- `SKNominalMatrixView<'a>` — a row-major matrix of borrowed labels: `&'a [&'a [&'a str]]`
  (outer slice = rows, inner slice = that row's column labels), with `shape()`,
  `number_of_rows()`, `number_of_columns()`, `value(row, column) -> &'a str`, and `is_empty()`.
- `SKCategoricalView<'a>` — `#[non_exhaustive] enum { Nominal(SKNominalMatrixView<'a>),
  Integer(ArrayView2<'a, i64>) }`, with `representation()` and once-per-operation dispatch.

The universal seam is `TryInto<SKCategoricalView<'a>, Error = SKError>`, provided by blanket
`From`/`TryFrom` impls for the concrete variants (e.g. `From<&'a [&'a [&'a str]]>`,
`From<ArrayView2<'a, i64>>`).

*Why a separate view instead of widening `SKDataView`?* The numeric view is generic over
`F: SKFloat`; category codes and text are not floats, and forcing them through `F` would be
lossy for large codes and would blur the meaning of every numeric-only algorithm's dense path.
A sibling enum keeps both boundaries precise.

*Alternatives considered:* (a) widen `SKDataView` with a `Categorical` variant — rejected, it
breaks the `F`-generic contract and every `as_dense` consumer; (b) accept `&[&[&str]]`
directly on the encoder — rejected, it forks a private seam instead of the shared one.

### Decision 2 — Missing-code sentinel is a documented constant `-1`

Integer-coded columns use `SK_CATEGORICAL_MISSING_CODE: i64 = -1` as the missing sentinel. It
is never registered as a category and always encodes to an all-zero column block. Kept as a
named constant so encoders and imputers agree.

*Why not `NaN` or `i64::MIN`?* There is no numeric NaN in an `i64` column, and `i64::MIN` is a
valid (if unlikely) code; `-1` matches scikit-learn's `-1` missing convention and is the
documented product convention.

### Decision 3 — Per-column category tables: `SKLabelTable` for text, a code table for integers

The fitted model stores one `SKOneHotColumnCategories` per input column:

- `Nominal(SKLabelTable)` — built with `sk_canonicalize_labels` (first-occurrence order,
  reversible), reusing the existing interning foundation.
- `Integer(SKIntegerCategoryTable)` — `codes: Vec<i64>` (index → code, first-occurrence order)
  plus `index_of: HashMap<i64, usize>`.

Both expose `number_of_categories()`; encode dispatch reads the matching variant once.

*Why not stringify integer codes and reuse `SKLabelTable` for both?* It would make `label_of`
return `"3"` (lossy-looking, and forcing a parse to recover the code). A tiny integer table is
cheaper and keeps the code exact.

### Decision 4 — Dense and sparse outputs are two methods; the builder picks the default

Model API: `transform_dense(...) -> Array2<F>`, `transform_sparse(...) -> CsMat<F>` (sprs),
and `transform(...) -> SKEncodedOutput<F>` (`#[non_exhaustive] enum { Dense(Array2<F>),
Sparse(CsMat<F>) }`) which dispatches on the builder's `sparse_output`. `SKPolynomialFeatures`
mirrors this with `SKExpandedOutput<F>`.

*Why an output enum rather than a generic output?* The associated-type transformer trait
(`SKFeatureTransformer`) is bound to numeric `SKDataView` input, which categorical data cannot
satisfy; an enum keeps the return type concrete and non-breaking (non-exhaustive) while still
letting the model be shared and its outputs converted by the consumer.

### Decision 5 — Parallel, two-pass CSR build

Sparse output is built in two passes: (1) per-row nonzero counts computed in parallel (rayon)
to derive the CSR row offsets; (2) a parallel fill of the value/index arrays, then
`CsMat::new_from_unsorted`. For one-hot each row has at most `number_of_columns` stored units;
for polynomial features a term is stored only when nonzero.

*Why `new_from_unsorted`?* It avoids a second sort while still guaranteeing a valid CSR; the
indices are produced in increasing column order per row by construction, so insertion is
deterministic.

### Decision 6 — Polynomial term layout and lazy generation

Terms are generated as exponent vectors in a documented order: bias first (when included),
then increasing total degree, then lexicographic within a degree; `interaction_only` restricts
exponents to `{0, 1}`; `degree` is the maximum total degree. The generator yields terms lazily
during transform so no per-term owned matrices are allocated; the model stores the term list
(it is `O(number_of_terms)` exponent vectors, not data).

*Why store terms but generate lazily?* The term list is metadata (small exponent vectors);
materializing *term values* per row is what must be streamed, and that is done row by row with
`azip!`/`map`. This satisfies the "lazy generator" intent of the wave plan.

### Decision 7 — `sprs` becomes a production dependency

Promote `sprs` to `sciencekit_preprocessing` production dependencies and to the workspace pins.

*Why now?* Both transformers emit sparse output; keeping the sparse build in `sciencekit_math`
or duplicating it would be worse. `sprs` is already the library's mandated sparse representation
(PRD §3, `data-view-boundary`).

### Decision 8 — Streaming via concatenation semantics

- `SKOneHotEncoder::fit_streaming` consumes an iterator of `SKCategoricalView` batches and
  derives tables in first-occurrence order across the concatenation (equivalent to dense fit
  on the concatenated rows), with a column-count consistency check per batch.
- `SKPolynomialFeatures::fit_streaming` consumes an `SKLazySource`; under an in-memory intent
  it collects and reuses the dense path, otherwise it learns the feature count from the first
  non-empty batch (the expansion shape depends only on the count, not on the values).

*Why is polynomial fit streaming trivial?* Its fit learns only the input feature count; no data
accumulation is needed, so streaming never has to buffer values.

### Road not taken — no `phf`/runtime perfect hash

The wave plan phrases the one-hot table as "interned string table, perfect-hash". A *runtime*
perfect-hash builder is not provided by `phf` (which is compile-time codegen) and would add a
dependency plus unsafe for a gain that a `HashMap`-backed interning table already delivers
(average `O(1)` lookup). The design realizes the plan's intent as a **deterministic, compact,
ordered interning table** (`SKLabelTable`, first-occurrence order) — stable across runs and
exportable. Recorded here as an explicit road not taken.

## Risks / Trade-offs

- **[Combinatorial blow-up in `SKPolynomialFeatures`]** the number of terms grows as
  `C(n + degree, degree)` → mitigated by validating `degree ≥ 1`, documenting the growth, and
  never materializing term values (only exponent metadata); a future change may add a guard or
  a sparse-first default for large `n`.
- **[Categorical view grows the shared crate]** touching `data-view-boundary` is cross-cutting →
  mitigated by keeping it additive and non-exhaustive (no breaking change to `SKDataView`) and
  by reusing the existing `TryInto`/rejection patterns verbatim.
- **[Dense/sparse divergence]** two code paths could drift → mitigated by making the sparse
  path structurally derived from the same per-row indicator source and by an equivalence test
  (densified sparse equals dense, exactly).
- **[Integer code as `F` output]** the indicator values are the model scalar `F`; for `f32`
  the unit is exact, so no precision concern.
- **[`handle_unknown(Ignore)` silent zeros]** can hide data quality issues → mitigated by the
  `Reject` default and a dedicated test.

## Migration Plan

Not applicable — additive change (new types, new estimators, new production dependency). No
existing public behavior changes; `SKDataView` and its consumers are untouched.

## Open Questions

None blocking. Deferred (do not change specs/approach): whether to add `min_frequency`/`drop`
to `SKOneHotEncoder` and a sparse-input path for `SKPolynomialFeatures` — both are additive
future changes.
