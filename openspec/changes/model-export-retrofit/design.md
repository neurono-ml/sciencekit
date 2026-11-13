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

### Decision 5a — Round-trip tolerance is zero

Import is lossless: the state carries every floating value verbatim, so an imported model
reproduces the original's `predict`/`transform`/`predict_labels` output **bit-for-bit** for
the same format version and input. The documented tolerance is therefore `0` (exact
equality), which is stronger than a numeric epsilon; the family tests assert `assert_eq!` on
the outputs rather than an approximate comparison. Benign NaN markers are the only values
compared by classification (`sk_states_equal` treats two NaNs as equal).

### Decision 5b — Explicit import limits

The state is sufficient to rebuild: every preprocessing model, imputer/indicator, linear
model, neighbour, SVM, tree, forest, exact/histogram booster, clusterer, decomposition and
outlier detector. These limitations are typed `UnsupportedOperation` errors with guidance,
never a partial model:

- Bagging ensembles prefix each member's tensors but do not carry the member's own metadata.
- Grid-search and successive-halving states carry the refitted winner's tensors and selection
  metadata, not the winner's metadata or candidate configuration.
- The typed pipeline's configured stage chain holds live estimator values; its nested stage
  states are still exported for inspection and per-stage import.
- Graph and streaming pipelines erase their fitted stage types behind `Any`/trait objects and
  cannot recover their concrete states; feature-union member states export, but the configured
  member chain cannot be rebuilt from data.

### Decision 5c — Honest coverage table

The change covers **47 fitted model types**. Export is universal (all 47 implement
`SKExportable`); native import is conditional. This is the exact reading of the
`model-export-coverage` spec: "Every fitted model exports state" is satisfied by the export
surface, "Native import where reconstructible" is satisfied by the 35 exact round-trips plus
the 10 typed refusals, and "Unsupported model is explicit" is satisfied because every refusal
is `UnsupportedOperation` with guidance and no partial model is ever produced.

| Model family / model type | Export | Import |
|---|---|---|
| The 35 exact-round-trip types: standard/min-max/robust scalers, one-hot encoder, polynomial features, simple imputer, missing indicator, KNN imputer, linear regression, ridge, lasso, elastic net, logistic regression, SGD classifier and regressor, KNN classifier and regressor, SVC, SVR, LinearSVC, decision tree classifier and regressor, random forest classifier and regressor, gradient boosting classifier and regressor, histogram gradient boosting classifier and regressor, KMeans, MiniBatchKMeans, DBSCAN, PCA, IncrementalPCA, TruncatedSVD, IsolationForest | yes | yes — bit-identical output (tolerance `0`) |
| `SKBaggingClassifierModel` / `SKBaggingRegressorModel` | yes | no — typed `UnsupportedOperation`: member metadata is not embedded |
| `SKGridSearchCVClassifierModel` / `SKGridSearchCVRegressorModel` | yes | no — typed `UnsupportedOperation`: winner tensors and selection metadata, not the winner configuration |
| `SKHalvingGridSearchClassifierModel` / `SKHalvingGridSearchRegressorModel` | yes | no — typed `UnsupportedOperation`: same winner-configuration gap |
| `SKPipelineClassifier` / `SKPipelineRegressor` (typed pipeline) | yes | no — typed `UnsupportedOperation`: configured stage chain holds live estimator values |
| `SKFeatureUnionModel` (static tuple) | yes | no — typed `UnsupportedOperation`: configured member chain holds live estimator values |
| `SKPipelineGraphModel` | no — typed `UnsupportedOperation`: stages behind `Any` boxes | — |
| `SKFittedStreamingPipeline` | no — typed `UnsupportedOperation`: type-erased fitted stages | — |

Totals: 35 exact round-trip; 10 composites export-only; 2 type-erased pipelines refuse at
export. No document claims that "every model round-trips".

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
