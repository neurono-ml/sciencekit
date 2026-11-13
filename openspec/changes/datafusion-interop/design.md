# Design

## Context

See `proposal.md` — Why. `interop-foundation` defines the tabular source contract, the
async-to-sync bridge (dedicated thread + current-thread Tokio runtime + bounded channel), and the
streaming provider adapter. `arrow-interop` converts Arrow record batches. DataFusion's
`DataFrame::execute_stream().await` returns a `SendableRecordBatchStream` (a pinned
`Stream<Item = Result<RecordBatch>> + Send`).

The existing streaming driver is synchronous and expects `SKLazySource::next_batch`. It must keep
working unchanged.

## Goals / Non-Goals

**Goals**

- Consume DataFusion results incrementally, with pushdown, without full collection.
- Keep the caller's async runtime free; keep CPU work on the compute pool.
- Deterministic row order across partitions.
- Reopenable sources for multi-epoch estimators.

**Non-Goals**

- No SQL parsing or plan construction beyond projection/filter forwarding.
- No custom `TableProvider` for arbitrary sciencekit estimators (only data registration).
- No distributed DataFusion (Ballista) support.
- No Arrow Flight or object-store credentials management.

## Decisions

### Reopen by cloning the DataFrame

A DataFusion `DataFrame` is a plan description and can be re-executed. A reopenable source stores a
cloned `DataFrame` plus the projection/filter and calls `execute_stream` on each pass. A one-shot
source stores the stream handle.

### Bridge at the foundation, not in this adapter

The async bridge is generic over an async stream of `Result<RecordBatch, E>`; this adapter only
wraps DataFusion's stream. That keeps cancellation, bounded buffering, and thread ownership in one
tested place. The bridge is sized from `SKExecutionPlan` buffer depth, defaulting to one.

### Projection and filter pushdown

`DataFrame::select_columns` (or `select`) applies the feature/target projection in declared order;
`DataFrame::filter` is applied when the caller supplies a predicate expression. Both happen before
`execute_stream`, so DataFusion's optimizer pushes them into scans (Parquet projection and row-group
pruning included).

### Ordering across partitions

DataFusion's `execute_stream` merges partition streams; the adapter records the batch sequence
produced by the stream and never reorders. Tests compare against a single-partition execution.
`maintain_order` semantics are documented as "stream order", not "sorted by any key".

### Registration back into DataFusion

Dense blocks register through `MemTable::try_new(schema, vec![vec![record_batch]])`; CSR blocks
register as a table of `(row, column, value)` plus a shape metadata field, because DataFusion has no
native sparse array type. This is an export convenience, not a storage format.

## Risks / Trade-offs

- **Async bridge adds a thread and a channel per source** → Mitigation: sources are per fit and
  short-lived; the thread is joined on drop; document the cost.
- **`DataFrame` clone may not be cheap for large plans** → Mitigation: clone once per pass, not per
  batch; expose a one-shot mode for single-pass estimators.
- **Partition order may differ from a naive expectation** → Mitigation: spec defines order as
  stream order and tests assert equality with the single-partition result.
- **DataFusion version coupling** → Mitigation: pin a supported range; Arrow versions aligned with
  the Arrow adapter.
- **Cancellation races** → Mitigation: channel close is the single cancellation signal; the reader
  checks it between polls; tests drop at several points.

## Migration Plan

Additive and feature-gated. No existing code changes.

## Open Questions

- Whether to expose a `SessionContext`-level convenience (`read_parquet_as_source`) or require the
  caller to build the DataFrame. Deferred: caller builds it, adapter only consumes.
