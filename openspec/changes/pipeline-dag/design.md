# Design

## Context

See `proposal.md` — Why. `type-safe-pipeline` supplies `SKFeatureBlock`, the stage adapter
and linear pipeline semantics. `petgraph` supplies the graph structure. The DAG executes
the same block medium; the difference is runtime branching and merging.

## Goals / Non-Goals

**Goals:**
- A validated DAG with deterministic topological execution.
- Reuse the linear stage adapter and block medium.
- Clear error reporting by node id.

**Non-Goals:**
- Compile-time graph typing.
- Streaming graphs and GPU execution.
- `memmap2` and export.

## Decisions

### Decision 1 — `petgraph` DiGraph with node ids

Nodes hold a transformer stage or the final estimator; edges represent block flow. The graph
is built through an `SKPipelineGraphBuilder` that returns node ids; `connect(from, to)`
adds edges. `validate` checks acyclicity, one final node and reachability.

### Decision 2 — Topological execution

`petgraph::algo::toposort` produces the order. Nodes at the same topological level are
processed in ascending node-id order for determinism. Each node reads its incoming blocks,
merges them in edge order (a node with one input is the common case; a merge node is
explicit), transforms and stores the output by node id.

### Decision 3 — Merge semantics

A node with multiple incoming edges receives a concatenation of the incoming blocks in edge
insertion order (the same representation policy as `SKFeatureUnion`). A node with zero
incoming edges is an input node and receives the caller's block.

### Decision 4 — Compatibility

The graph is generic over `F: SKFloat`; every node shares that scalar by construction. Edge
representation compatibility is checked at fit time by the receiving stage; its
`UnsupportedRepresentation` is wrapped with the node id.

### Decision 5 — Final estimator and prediction

Exactly one final node holds the supervised estimator. After fitting, the graph stores the
final model; classifier/regressor wrappers delegate prediction after executing the
transformers. The graph model is immutable and `Send + Sync`.

### Decision 6 — File layout

```
crates/sciencekit_pipeline/src/graph/
  mod.rs, builder.rs, node.rs, validate.rs, execute.rs, model.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Runtime typing loses compile-time guarantees** → validation and clear node-id errors;
  the linear typed pipeline remains the safe default.
- **Merge order can surprise** → edge insertion order is documented and tested.
- **Graph execution is sequential per level** → parallelism can be added per level without
  changing semantics.
- **`petgraph` is a new dependency** → small and widely used; MSRV compatible.

## Migration Plan

Additive: new graph module in `sciencekit_pipeline`. No existing API changes.

## Open Questions

None blocking. Streaming graphs and parallel level execution are future changes.
