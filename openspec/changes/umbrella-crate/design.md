# Design

## Context

See `proposal.md` — Why. The workspace has many `sciencekit_*` crates with a stable public
surface. The umbrella crate is a thin re-export layer; it must not add logic or create
cycles.

## Goals / Non-Goals

**Goals:**
- One dependency for users.
- A small prelude.
- Optional heavy integrations behind features.

**Non-Goals:**
- Changing sub-crate APIs.
- Bundling Python bindings.
- A second version line.

## Decisions

### Decision 1 — Module-per-family re-exports

The umbrella exposes modules such as `preprocessing`, `linear_model`, `neighbors`, `svm`,
`tree`, `ensemble`, `cluster`, `decomposition`, `outlier`, `metrics`, `model_selection`,
`pipeline` and `interop`, each re-exporting its crate's public items. Common types are
re-exported at the root.

### Decision 2 — Prelude

The prelude re-exports the builder/fit/predict/transform traits, `SKDataView`,
`SKTargetView`, `SKError`, `SKFloat`, the execution mode/backend and the most-used
estimators/builders. It does not re-export every model to keep compile times reasonable.

### Decision 3 — Feature flags

`interop`, `safetensors`, `json`, `onnx`, `polars`, `datafusion`, `arrow` and `parquet`
map to the corresponding optional crates/features. The default build is core-only.
Python is not a feature; it is a separate `sciencekit_python` crate.

### Decision 4 — Documentation

Crate-level docs explain installation, the prelude, the family modules and the mdBook/repo
links. docs.rs metadata enables the `docsrs` cfg and feature badges.

### Decision 5 — File layout

```
crates/sciencekit/src/
  lib.rs
  prelude.rs
  preprocessing.rs
  linear_model.rs
  neighbors.rs
  svm.rs
  tree.rs
  ensemble.rs
  cluster.rs
  decomposition.rs
  outlier.rs
  metrics.rs
  model_selection.rs
  pipeline.rs
  interop.rs
```

Each module is a re-export-only file; no logic.

## Risks / Trade-offs

- **Re-export name collisions** → explicit `pub use` paths and compile tests.
- **Compile time grows with the umbrella** → feature-gated heavy integrations and a small
  prelude.
- **Docs duplication** → crate-level docs point to the mdBook instead of repeating it.
- **Versioning** → the umbrella follows the workspace version.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking.
