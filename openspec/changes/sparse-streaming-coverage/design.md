# Design

## Context

See `proposal.md` — Why. `sparse-streaming-batches` delivered the owned
`SKFeatureBlock::Sparse` block and the prefetch executor. Some estimators use
`as_dense()` in their streaming update step, which rejects or densifies sparse batches.
This change audits and fixes those paths.

## Goals / Non-Goals

**Goals:**
- A truthful dense/sparse coverage matrix.
- Sparse streaming for every estimator that claims streaming.
- No densification and bounded memory.

**Non-Goals:**
- New algorithms or `memmap2`.
- GPU streaming.
- Changing the streaming executor.

## Decisions

### Decision 1 — Coverage matrix

Each `fit_streaming`/`partial_fit` implementation is audited and classified:
`DenseOnly`, `SparseOnly` (unlikely) or `Both`. The matrix lives in the crate docs and is
enforced by tests. A dense-only streaming estimator must reject CSR with
`UnsupportedRepresentation`.

### Decision 2 — Shared batch dispatch

Where an algorithm already has a dense/sparse design abstraction (linear models,
preprocessors), the streaming update uses the same abstraction instead of `as_dense()`.
For scalers/encoders/imputers, the update uses sparse reductions/entry scans; the retained
state stays small and dense.

### Decision 3 — Sparse equivalences

Every sparse streaming test compares against the dense streaming path on the same logical
matrix, covering one batch, multiple batches, multi-epoch and `partial_fit`.

### Decision 4 — Memory

The sparse path never materializes a dense `n × d` block; the only dense objects are
per-feature statistics/coefficients and the current batch. Tests use a wide sparse matrix
and assert no dense conversion.

### Decision 5 — File layout

Per crate, the streaming module gains `sparse_*_tests.rs` and, where needed, a sparse
update path. A workspace-level coverage test enumerates the matrix.

## Risks / Trade-offs

- **Some algorithms are inherently dense in streaming** → documented exception and error.
- **Sparse reductions can be slower** → correctness first; benchmark in acceptance.
- **Coverage drift** → a workspace test that requires every streaming estimator to declare
  its coverage.

## Migration Plan

Additive: sparse paths and tests. Dense behavior unchanged.

## Open Questions

None blocking.
