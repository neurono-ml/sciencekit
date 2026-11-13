# Proposal

## Why

Linear pipelines cover the common case, but production feature graphs branch and merge
(numeric and text branches feeding a single estimator). The PRD asks for DAGs via
`petgraph`; the linear typed pipeline cannot express them without a graph layer.

## What Changes

- New `SKPipelineGraph` in `sciencekit_pipeline`: a `petgraph`-backed directed acyclic
  graph of stages over the shared `SKFeatureBlock` medium.
- Node kinds: transformer, feature union, and a final estimator node; edges carry the
  feature block flow.
- Runtime validation: cycle detection, exactly one final estimator, every non-root node
  reachable, scalar compatibility (all nodes share `F`) and representation compatibility
  at each edge.
- `fit` executes in topological order, fitting each node on its incoming block and
  transforming it forward; `predict` reuses the fitted graph.
- Deterministic topological execution and error reporting with node ids.
- Accessors: node count, edge count, topological order, fitted node blocks and the final
  model.
- Explicit non-goals: compile-time graph typing, streaming graphs, GPU, `memmap2` and
  export.
- Companion tests: diamond graphs, disconnected/cyclic graphs, missing final node,
  hand-checkable fit/predict, dense/sparse flow, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `pipeline-graph`: DAG construction, validation, topological fitting/prediction,
  deterministic execution and graph accessors.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_pipeline` gains the graph modules; a new `petgraph`
  dependency.
- **APIs:** additive.
- **Dependencies:** `petgraph`, `sciencekit_common`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** gives production feature graphs a runtime DAG executor while the typed
  linear pipeline remains the compile-time-safe path.
- **Acceptance criteria:** small and large graphs, concurrency, companion `*_tests.rs`
  modules. Export is deferred to Wave 7.
