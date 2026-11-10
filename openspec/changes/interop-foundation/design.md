# Design

## Context

See `proposal.md` — Why. The existing boundary (spec `data-view-boundary`) is a zero-copy
`SKDataView<'a, F>` (dense row-major `ArrayView2` or CSR `CsMatView`) plus `SKTargetView` and
`SKCategoricalView`. Streaming (specs `streaming-batches`, `streaming-source-provider`) consumes
**owned** blocks via `SKLazySource` / `SKSupervisedLazySource` and a provider seam that can reopen a
pass. There is no crate or type for heterogeneous tables, and no interop crate exists yet.

The hard constraint is layout: DataFrames are **column-major and heterogeneous**; `SKDataView` is
**row-major and homogeneous**. A multi-column table cannot be aliased as a row-major matrix without
copying. The design must be honest about where zero-copy is possible.

## Goals / Non-Goals

**Goals**

- One table model and one conversion path shared by every engine and every language binding.
- Deterministic feature order, one target, one missing-value policy.
- Streaming blocks remain owned and thread-transferable.
- Async engines feed the sync driver without blocking the caller's runtime.
- Errors are precise and namespaced per column.

**Non-Goals**

- No estimator signature changes and no algorithm changes.
- No engine dependency in the core crate; concrete engines are optional features added later.
- No multi-target regression, no nested/struct columns as features, no temporal feature expansion.
- No schema inference rules beyond what an adapter supplies; inference belongs to the adapter.

## Decisions

### One feature-gated crate instead of satellite crates

`sciencekit_interop` owns the neutral core. Arrow, Polars, DataFusion, and memmap arrive as optional
features (`arrow`, `polars`, `datafusion`, `memmap`) added by their own changes. This matches PRD
§3.1 (`sciencekit_interop`) and avoids a crate explosion while keeping dependencies opt-in.
Alternative considered: one satellite crate per engine; rejected because every satellite would
depend on a shared core anyway and the workspace already has a single interop entry in the PRD.

### Column-major batch as the neutral in-memory shape

`SKTabularBatch` stores `Vec<SKColumn>` where `SKColumn` is typed and nullable-aware. It is the
natural shape for engines and binds cleanly to Arrow without committing the core to Arrow.

Materialization rules:

| Source shape | Target | Copy? |
|---|---|---|
| contiguous row-major `ArrayView2<F>` | `SKDataView::Dense` | no |
| single numeric column, contiguous | `SKDataView::Dense` row matrix | no (reshape) |
| multi-column table, any engine | `SKDataView::Dense` | yes (transpose/interleave) |
| CSR buffers | `SKDataView::Sparse` | no |
| multi-column table → streaming | `SKFeatureBlock::Dense` owned | yes |

The spec records this explicitly so "zero-copy" is never claimed for a copy that is structurally
unavoidable.

### Selection and roles are explicit, not inferred

`SKColumnSelection` names features (ordered) and at most one target; every other column is ignored.
Name resolution happens against the schema before reading rows. Duplicate names are rejected unless
the selection is positional. Alternative: infer target as "last column" — rejected because it
silently mislabels data.

### Missing values are a policy, not a per-engine default

- `Reject` — any null is an error.
- `NumericNaN` — float nulls become NaN; integer columns with nulls are promoted to the model
  scalar first (never truncated); NaN is documented as missing for estimators.
- `CategoricalMissingCode` — nominal nulls become `SK_CATEGORICAL_MISSING_CODE`.
- `ConstantFill(value)` — numeric replacement, explicit and visible.

Default is `Reject`, so a caller must acknowledge nullability. Alternative: silently map to NaN —
rejected because it hides data quality problems and corrupts categorical encoders.

### The tabular source is a provider, not a new streaming trait

`SKTabularSource` yields `SKTabularBatch` and can be opened per pass. A small adapter converts each
batch to `SKDataBatch` / `SKSupervisedBatch` and exposes `SKSourceProvider` /
`SKSupervisedSourceProvider`. The streaming executor is untouched; the one-epoch/multi-epoch rules
from `streaming-source-provider` still apply. Alternative: add a second executor for tabular
sources — rejected for duplicate logic.

### Async bridge on a dedicated thread

Async engines return `SendableRecordBatchStream` or an async `CollectBatches`. The bridge owns a
std thread running a current-thread Tokio runtime, polls the stream there, and sends owned batches
through a bounded channel sized by the plan's buffer depth. The consumer side implements the sync
`SKLazySource`. Dropping the consumer closes the channel, the reader observes it, and the runtime is
dropped. This keeps Tokio's I/O role and rayon's compute role separate (PRD §5) and never parks an
async worker.

### Errors extend the central taxonomy

New variants in `SKError` (spec `error-model`): `UnknownColumn`, `DuplicateColumn`,
`UnsupportedColumnType`, `NullValueInColumn`, `UnevenColumnLengths`, `SchemaMismatch`. They carry
the column name and a suggestion. Estimators keep returning `SKError`; no interop-specific error
type leaks.

## Risks / Trade-offs

- **Copy on every multi-column batch** → Mitigation: copy only per batch, bounded by the streaming
  block size; document it; a future columnar view variant can remove it without breaking this seam.
- **NaN as missing conflicts with legitimate NaN values** → Mitigation: `Reject` is the default;
  `NumericNaN` must be chosen explicitly; missing-indicator estimators can exploit it.
- **Integer promotion above 2^53 loses precision** → Mitigation: reject integer nulls when the
  scalar cannot represent the column's magnitude; document the check.
- **Bounded channel sizing interacts with the plan** → Mitigation: the adapter accepts the depth
  from the plan and defaults to one (double buffering); tests assert at most two batches resident.
- **Feature flags multiply CI combinations** → Mitigation: CI builds the core alone plus each
  feature, and the workspace default build stays dependency-light.

## Migration Plan

Additive: new crate, new types, new error variants (`#[non_exhaustive]` already covers consumers).
No existing call site changes. Concrete adapters land in later changes and are feature-gated.

## Open Questions

None blocking the spec: exact module split inside the crate (schema, selection, columns, conversion,
source, bridge) follows the 200-line rule as the implementation lands.
