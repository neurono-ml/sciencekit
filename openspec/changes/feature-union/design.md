# Design

## Context

See `proposal.md` — Why. `type-safe-pipeline` supplies `SKFeatureBlock`, the stage adapter
and the linear pipeline. The union is a single stage that fans the same input to several
members and concatenates their outputs.

## Goals / Non-Goals

**Goals:**
- Heterogeneous member types with compile-time composition.
- Correct dense/CSR concatenation without densifying sparse inputs.
- Reuse inside `SKPipeline`.

**Non-Goals:**
- Column-name transforms, weights, drop/remainder policies.
- DAG unions and streaming unions.
- `memmap2` and export.

## Decisions

### Decision 1 — Tuple-based members

`SKFeatureUnion` is generic over a tuple of member transformers. A
`SKFeatureUnionStages<F>` trait implemented for tuples up to eight members (via a small
macro) fits each member and returns a tuple of fitted models. This keeps each member's
concrete type and avoids boxing for the common case.

A boxed dynamic list (`Vec<Box<dyn SKBlockTransformer<F>>>`) is offered as a documented
fallback for runtime construction; it uses the same `SKFeatureBlock` medium and loses only
the per-member static type.

### Decision 2 — Fit and transform

Fit maps the tuple, fitting every member on the same borrowed block. Transform maps the
fitted tuple, transforms the block with every member, then concatenates the outputs in
member order.

### Decision 3 — Concatenation

- All dense: `ndarray::concatenate(Axis(1), views)`.
- Any sparse: build a CSR output by horizontally stacking each member; a dense member is
  converted to CSR (a local conversion, not a densification of sparse input), a sparse
  member contributes its stored entries with a column offset.

The output width is the sum of member widths; row counts must match.

### Decision 4 — Errors and accessors

A member error is wrapped with its index (or surfaced with context). Shape mismatches are
detected before writing the output. The fitted union stores each member's width and exposes
the total feature count.

### Decision 5 — Pipeline integration

The union implements `SKPipelineTransformer` over `SKFeatureBlock`, so it can be a stage in
the typed pipeline. It does not implement the old `SKFeatureTransformer` directly because
its input is the block medium.

### Decision 6 — File layout

```
crates/sciencekit_pipeline/src/feature_union/
  mod.rs, builder.rs, stages.rs, concatenate.rs, model.rs, boxed.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Tuple arity is bounded** → eight members covers the common case; the boxed fallback
  handles larger dynamic unions.
- **Sparse concatenation copies stored entries** → inherent to building a new CSR; no
  densification.
- **Member errors can be deep** → wrap with the member index.
- **Column names are absent** → a later change can add a named union without changing the
  block medium.

## Migration Plan

Additive: new stage in `sciencekit_pipeline`. No existing API changes.

## Open Questions

None blocking. Named columns and streaming unions are future changes.
