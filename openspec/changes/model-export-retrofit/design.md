# Design

## Context

See `proposal.md` — Why. `interop-contracts` defines `SKExportable<F>`,
`SKImportable<F>`, `SKModelState<F>`, `SKTensor<F>` and metadata. Every model crate already
depends on `sciencekit_common`, so implementations are additive and local to each model.

## Goals / Non-Goals

**Goals:**
- A uniform, documented state for every fitted model family.
- Round-trip prediction for models whose state is reconstructible.
- One shared test harness so families cannot silently drift.

**Non-Goals:**
- File formats (Safetensors/JSON/ONNX) and Python bindings.
- Automatic derive macros.
- Cross-version migration beyond the documented format version.

## Decisions

### Decision 1 — Export lives beside the model

Each crate adds `model/export.rs` (and `model/import.rs`) implementing the common traits
for its model types. This keeps private fields accessible without exposing them and keeps
the format crates free of model-specific code.

### Decision 2 — Stable dotted tensor names

Names follow `<scope>.<field>`:

```
standard_scaler.mean
standard_scaler.scale
minmax_scaler.data_min
minmax_scaler.data_max
one_hot.categories.<column>
linear.coefficients
linear.intercept
tree.nodes.feature
tree.nodes.threshold
tree.nodes.left
tree.nodes.right
tree.nodes.value
forest.trees.<index>.nodes.*
svm.support_vectors
svm.dual_coefficients
svm.bias
kmeans.cluster_centers
pca.components
pca.mean
pipeline.stage.<index>.*
```

The exact names are recorded per family in the family task. Names are part of the
compatibility promise; renaming requires a format-version bump.

### Decision 3 — Metadata carries configuration

Metadata includes estimator name, format version, scalar dtype, feature count, class/label
tables, backend, solver and the hyperparameters needed to rebuild the configuration.
Hyperparameters use `SKParameterValue`.

### Decision 4 — Import reconstructs from configuration + state

`SKImportable` reads the metadata into the model's configuration, validates the tensor
names/shapes/dtypes, constructs the model and verifies the feature/class dimensions. A
model whose state cannot reconstruct the configuration returns `UnsupportedOperation`.

### Decision 5 — Shared test harness

A dev-only helper in `sciencekit_common` (or a small `sciencekit_test_support` module)
provides `assert_export_round_trip(model, features)`: export, import, compare predictions
within tolerance; assert non-empty state and stable names. Each family calls it with its
fixtures.

### Decision 6 — File layout

Per crate, a `model/export.rs` and optional `model/import.rs`; the family's `mod.rs`
declares and re-exports nothing public beyond the trait impls. The shared helper lives in
`crates/sciencekit_common/src/interop/testing.rs` behind `#[cfg(feature = "test-support")]`
or a dev-only module.

## Risks / Trade-offs

- **Large mechanical change** → split tasks by family; the shared harness catches drift.
- **Stable names are hard to change later** → document them and version the format.
- **Some models cannot be reconstructed** → explicit `UnsupportedOperation`.
- **Private state access** → export lives beside the model in its own crate.
- **Model variants (single/multi-alpha, solver paths)** → metadata records the variant and
  import validates it.

## Migration Plan

Additive trait impls. No existing API changes. A model without an export impl is a
compile-time gap only for consumers that require `SKExportable`.

## Open Questions

None blocking. Derive macros and cross-version migration are future conveniences.
