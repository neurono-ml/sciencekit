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

### Decision 0 — The umbrella lives in the existing workspace-root package

The plan said "create `crates/sciencekit`". The workspace already has a root
package named `sciencekit` occupying the umbrella slot (an empty `src/lib.rs`
plus the workspace-wide export-coverage tests), and Cargo cannot host a second
package with the same name in one workspace. The umbrella is therefore
implemented in the **existing root package** (`Cargo.toml` `[package]` +
`src/`): the same public outcome, one fewer package, and no rename of the
published crate.

Consequences: the family modules live in `src/<family>.rs` instead of
`crates/sciencekit/src/`, and the root `[dependencies]` (previously
`[dev-dependencies]`) are the sub-crates the umbrella re-exports.


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

`opencl`, `telemetry`, `onnx`, `polars`, `datafusion`, `memmap` and `parquet` map to
the corresponding optional crates/features through `dep:` forwarding, so the
umbrella never re-declares a dependency version. The default build is core-only.
Python is not a feature; it is a separate `sciencekit_python` crate.

The forwarding is complete for every sub-crate switch that changes what the
umbrella can *do*: `otlp-grpc`/`otlp-http` (one OTLP transport each),
`observability-export`, `compression-gzip`/`-brotli`/`-snappy` and
`allocator-jemalloc`/`-mimalloc`. `telemetry` forwards the sub-crate's own
`telemetry` umbrella feature (both transports); forwarding only the dependency
would enable the crate while leaving every transport compiled out, so the
pipeline builder would reject every configuration. The two allocator features are
mutually exclusive — one global allocator per process — so the crate documents
that `--all-features` is not a valid build and the docs.rs metadata lists the
integration features explicitly instead. Three switches stay out on purpose:
`blas-backend` needs a system BLAS at link time, and `gpu-cuda` / `gpu-rocm`
have no implementation to select yet.

The family modules are gated the same way: `telemetry` exists only with its
feature, the ONNX re-exports only with `onnx`, the engine adapters only with
their adapter feature, and the `sciencekit_gpu_opencl` re-exports only with
`opencl`. A compile test asserts the mirror in whichever combination is built.

### Decision 3b — Collisions are resolved by family, never by flattening

Three distinct `SKClassWeight` policies (tree/ensemble, linear-model logistic,
SVM) and two `sk_weighted_average` functions (neighbors, impute) share names
across families. Each stays in its family module and is deliberately omitted
from the prelude. A quieter third collision is recorded too: `SKSplitter` is a
split-selection enum in `tree` and the train/test partition trait in
`model_selection`; neither is a prelude name. Names that are the *same* type
re-exported by several modules (`SKFeatureBlock`, the device vocabulary,
`SKOutlierLabel`, `SKKernel`) are safe to duplicate and are documented as one
identity; `reexports_tests.rs` proves the identity by assigning between the two
spellings and each helper only type-checks when both paths name one type.

### Decision 4 — Documentation

Crate-level docs explain installation, the prelude, the family modules and the mdBook/repo
links. docs.rs metadata enables the `docsrs` cfg and feature badges.

### Decision 5 — File layout

```
src/
  lib.rs                  # crate docs + shared vocabulary at the root
  prelude.rs
  math.rs
  preprocessing.rs
  impute.rs
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
  interop.rs              # includes the feature-gated ONNX re-exports
  accelerators.rs
  telemetry.rs            # feature-gated
  ffi.rs
  features_tests.rs
  prelude_tests.rs
  reexports_tests.rs
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
