## Why

The PRD promises native out-of-core fits, but the streaming block is **dense only**:
`SKDataBatch` and `SKSupervisedBatch` own an `Array2<F>`. A sparse design — the common shape
of text, recommendation and one-hot-encoded data — cannot be streamed at all: it must be
densified before it enters the stream, which destroys both the memory guarantee and the
zero-copy sparse path the in-memory `fit` already provides. This is recorded in the PRD's
future-improvements list.

The in-memory boundary (`SKDataView`) already models both representations and every
estimator already dispatches on it. The gap is the streaming vocabulary: it has no way to
carry a CSR block.

## What Changes

- **New owned block type** `SKFeatureBlock<F>` in `sciencekit_common`: `Dense(Array2<F>)` or
  `Sparse(CsMat<F>)`, non-exhaustive, with `view() -> SKDataView`, row/column counts and a
  byte-size estimate for the execution plan's memory guard.
- **`SKDataBatch` and `SKSupervisedBatch` carry a feature block** instead of an `Array2`,
  preserving their position, final-block flag and owned-data guarantees; the dense
  constructors keep working and sparse constructors are added.
- **The prefetch driver passes the representation through untouched**: no densification, no
  conversion, block order preserved, and the double-buffer memory guard accounts for the
  stored CSR arrays rather than `n × p`.
- **All numeric streaming estimators gain sparse batches**: the scalers, polynomial features
  and the simple imputer compute their column statistics over CSR rows; the linear models
  reuse the sparse fitting paths they already have for the in-memory `fit`.
- **A shared conformance suite** asserts, for every streaming estimator, that a sparse stream
  equals the dense stream and the dense `fit` on the concatenated data, and that no dense
  `n × p` copy is built.
- **`SKOneHotEncoder` stays outside**: its streaming seam carries categorical views, not
  numeric blocks; it is documented as the exception and remains unchanged.

## Capabilities

### New Capabilities

- `sparse-streaming-batches`: the owned feature block and the representation-generic batch —
  construction, accessors, invariants, memory accounting and the no-densification guarantee.

### Modified Capabilities

- `streaming-batches`: owned blocks carry either a dense matrix or a CSR matrix; the sources,
  the prefetch driver and the estimates that consume them preserve the representation
  end to end.

## Impact

- **New code:** `sciencekit_common/src/batching/feature_block/` (`mod`, `block`,
  `block_tests` + companion `*_tests.rs`) and the updated batch types.
- **Modified code:** every numeric `fit_streaming` implementation across
  `sciencekit_preprocessing`, `sciencekit_impute` and `sciencekit_linear_model`; the prefetch
  driver; the double-buffer byte accounting; `SKSGD*`'s sparse batch path.
- **Dependencies:** none new — `sprs` is already a dependency of `sciencekit_common`.
- **Public API:** **BREAKING** for `SKDataBatch::data()` / `SKSupervisedBatch::features()`,
  which are replaced by representation-aware accessors (`view()`, `number_of_rows`,
  `number_of_columns`, `is_sparse`); the dense constructors are preserved.
- **Sequencing:** lands after `streaming-source-provider`, whose provider seam is the clean
  place to add sparse blocks (a reopenable provider can yield sparse passes without touching
  the estimator signatures again).
- **Acceptance (PRD §8.7/§10.3):** a sparse stream equals the dense stream and the dense fit
  on the concatenated data; no dense `n × p` copy is built; the memory guard uses the stored
  CSR size; blocks of any size or order converge to the same model; both floats and
  concurrency are covered.
