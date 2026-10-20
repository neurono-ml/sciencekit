# Design

## Context

The seams this change plugs into already exist: `SKDataView` (dense `ArrayView2<F>` / sparse
`CsMatView<F>`) entered through the `TryInto` fallible bound, `SKUnsupervisedFit` returning a
distinct model, `SKFeatureTransformer` with an associated `Output`, the builder contract
(`backend` default `Faer`, `execution_mode` default `Automatic`), the observability span
runner (`sk_run_operation`) and the streaming driver (`sk_run_streaming_driver` over
`SKLazySource`) — the exact pattern `standard-scaler` and `minmax-and-robust-scalers` use.
`sprs` and `tdigest` are already workspace pins. See proposal.md for motivation and
`specs/simple-imputer/spec.md` / `specs/missing-indicator/spec.md` for the behavior contract.

## Goals / Non-Goals

**Goals:**
- A new `sciencekit_impute` crate that owns the missing-data estimators and follows the
  established folder-module, builder, span and streaming patterns character for character.
- Representation-preserving imputation (dense→dense, CSR→CSR) with sklearn's marker and
  empty-column semantics, plus the missingness mask.
- All three regimes (`fit`, `partial_fit`, `fit_streaming`) so the iterative mandate
  (naive → tests → performance → streaming) is not skipped.

**Non-Goals:**
- Multivariate imputation (`SKKNNImputer`, `IterativeImputer`) — Wave 3.
- Categorical/text imputation — belongs to the nominal view added by the previous change
  and will reuse its `SK_CATEGORICAL_MISSING_CODE`.
- Model export/serialization and metrics — deferred to the interop/metrics changes, as in
  `standard-scaler`.
- Umbrella re-export of the new crate — Phase 7 per the PRD.

## Decisions

### 1. Own crate, own workspace member
`crates/sciencekit_impute` added to `[workspace] members`, depending on
`sciencekit_common`, `sciencekit_math`, `sprs` and `tdigest`. *Why:* the PRD reserves the
name and the one-crate-per-change rule keeps the PR small. *Alternative rejected:* dropping
the estimators into `sciencekit_preprocessing` — convenient, but it contradicts the reserved
crate tree and mixes missing-data handling with feature engineering.

### 2. `SKImputedOutput<F>` enum owned by the new crate
The model's `SKFeatureTransformer::Output` is
`SKImputedOutput<F> { Dense(Array2<F>), Sparse(CsMat<F>) }`. *Why:* the wave plan demands
sparse-aware behavior, and an `Array2`-only output would force `as_dense` rejection — the
opposite of zero-copy. *Alternatives rejected:* reusing `SKEncodedOutput` (it is the
encoder's vocabulary-typed result living in another crate; renaming it to a shared
`SKOwnedMatrix` would touch merged code for no behavior gain) and returning borrowed views
(the transform fills cells, so the result cannot borrow the input).

### 3. Marker semantics, one per representation
Dense: a cell is missing iff `is_nan()` when the marker is `NaN` (the default), otherwise
iff `== marker`. CSR: explicit entries follow the same rule, and a `0.0` marker additionally
marks the implicit zeros (the sparse missingness idiom); any other marker leaves implicit
zeros observed. *Why:* matches sklearn, where `missing_values=0` on sparse is the
"I lost my dense matrix" idiom, while a `NaN` marker on sparse data must not silently treat
every structural zero as a hole. *Risk accepted:* with a `0.0` marker, genuine structural
zeros are imputed — documented in the spec scenario so the choice is explicit at the call
site.

### 4. Statistics engines
Columns are independent, so the dense path iterates columns with `rayon` `par_iter` and
fills cells with `azip!`/`map` — never manual index loops:
- **Mean** — compensated (pairwise) sum/count, reusing the math crate's reduction style;
  exact and order-independent enough for the deterministic test contract.
- **Median** — collect the observed values of the column and take the order statistic of
  the sorted slice. Exact; memory is O(observed values) per column. *Alternative rejected:*
  `tdigest` on the dense path — approximate where exact is affordable, and the dense tests
  pin exact values.
- **MostFrequent** — value→count map, tie broken toward the smallest key (sklearn rule);
  deterministic because the tie rule does not depend on insertion or iteration order.
- **Constant** — `fill_value` (default `0.0`), no state.

### 5. Incremental and streaming state mirrors the scalers
`partial_fit` owns the accumulating state on the model: count/sum for the mean, the value
counts for the mode, nothing for the constant, and a per-column `tdigest` sketch for the
median (exact under `Mean`/`MostFrequent`/`Constant`, as the spec requires; the median
approximation is confined to the sketch path). `fit_streaming` copies the
`standard-scaler` split: in-process intents collect the batches and reuse the dense path so
the result is exactly the concatenated fit; otherwise the streaming driver updates the
per-batch state. The same state records the columns that carried a missing cell, so
`partial_fit` and the driver rebuild the composed indicator's scope as batches arrive — an
incrementally fitted model stacks the same mask as the concatenated fit. Spans are built
from `SKOperationAttributes` carrying the model's backend kind, exactly as the scalers do.

### 6. The mask is typed on `F`
`SKMissingIndicator` emits `0.0`/`1.0` in `SKFloat` rather than `bool`. *Why:*
`SKFeatureTransformer` chains statically typed stages, and the stacked-with-`add_indicator`
output must be one matrix of one scalar type. *Alternative rejected:* a `bool` mask
(sklearn parity) — it would introduce a second scalar type through every downstream
pipeline; the deviation is recorded in the ADR and asserted by tests.

### 7. Composition instead of a special-cased transform
`add_indicator(true)` stores a `SKMissingIndicatorModel` inside the imputer model and the
transform builds `horizontally_stack(imputed, mask)` in a second pass. *Why:* the two
estimators stay independently testable and the stacking is a pure layout step.

### 8. Module layout
`sciencekit_impute/src/` holds `simple_imputer/` and `missing_indicator/` folder modules,
each a pure-dispatcher `mod.rs` over `builder.rs`, `core_implementation.rs`,
`model/{core_implementation,partial_fit,transform}.rs`, `streaming/{state,batches,fit}.rs`
and companion `*_tests.rs` declared at the end of the dispatcher — the `standard-scaler`
shape, no file past 200 lines.

## Risks / Trade-offs

- [Exact median holds every observed value of a column in memory] → only on the dense/collected
  path, which by definition already holds the matrix; the streaming driver uses the sketch.
  Documented in the spec's streaming scenario.
- [Mode count maps grow with distinct values per column] → memory is bounded by the column's
  cardinality, not its row count; the risk is called out here and in the docs, and a
  streaming cardinality sketch can replace it later without a spec change.
- [Sketch median ≠ exact median across regimes] → the tolerance test pins the driver result
  to the documented `tdigest` accuracy instead of demanding bit equality, while the
  in-memory intent path is asserted bit-equal to the dense fit.
- [`0.0` marker on sparse data imputes structural zeros] → opt-in and scenario-documented;
  the default `NaN` marker never touches implicit zeros.
- [`add_indicator` changes output width] → the width rule is a spec requirement with its own
  scenario, so consumers can rely on it.
- [A new crate grows CI time] → it is one more `-p sciencekit_impute` in the same workspace
  test/clippy run; no new workflow is needed.

## Migration Plan

Purely additive: no existing item changes, so no deprecation path. The change lands as one
code PR (crate + workspace member + tests), one openspec PR and one docs PR; the umbrella
re-export and the changelog entry land with the change as usual.

## Open Questions

None — the marker semantics, the output type, the median strategy across regimes and the
mask scalar type are all resolved above and pinned by the specs.
