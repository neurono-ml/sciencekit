# Design

## Context

See `proposal.md` — Why. `sciencekit_common` exposes `SKFeatureTransformer` (associated
output), `SKUnsupervisedFit`, `SKSupervisedFit`, the predictors and `SKDataView`. Existing
transformer models return `Array2`, `CsMat`, `SKImputedOutput` or `SKIndicatorOutput`.
`SKFeatureBlock` is the owned dense/sparse medium. `sciencekit_pipeline` is new.

## Goals / Non-Goals

**Goals:**
- A compile-time typed linear chain with a single `fit`/`predict` surface.
- A uniform owned block medium between stages.
- Dense/sparse flow without implicit densification.

**Non-Goals:**
- FeatureUnion (W6.7), DAGs (W6.8), streaming (W6.9).
- Runtime stage lists or trait-object pipelines.
- `memmap2` and export.

## Decisions

### Decision 1 — `SKFeatureBlock` is the pipeline medium

Every stage receives and returns an owned `SKFeatureBlock<F>` (dense `Array2` or CSR
`CsMat`). Existing transformer outputs are adapted through `From` impls:

- `Array2<F> → SKFeatureBlock`
- `CsMat<F> → SKFeatureBlock`
- `SKImputedOutput<F> → SKFeatureBlock` (implemented in the impute crate)
- `SKIndicatorOutput<F> → SKFeatureBlock` (implemented in the impute crate)

The first stage converts the caller's `SKDataView` into an owned `SKFeatureBlock` (one boundary copy, accepted in the ADR) and then borrows that block; subsequent stages use owned blocks. This
avoids copying the input at the pipeline boundary.

### Decision 2 — A stage adapter over the existing transformer contract

```rust
pub trait SKPipelineTransformer<F: SKFloat> {
    type Fitted;
    fn fit_stage(&self, block: &SKFeatureBlock<F>) -> Result<Self::Fitted, SKError>;
    fn transform_stage(&self, fitted: &Self::Fitted, block: &SKFeatureBlock<F>) -> Result<SKFeatureBlock<F>, SKError>;
}
```

A blanket implementation covers any `T: SKUnsupervisedFit<F>` whose model implements
`SKFeatureTransformer<F>` and whose `Output: Into<SKFeatureBlock<F>>`. This reuses every
existing transformer without changing its trait.

### Decision 3 — Typed builder with nested tuples

`SKPipelineBuilder<F, Stages>` uses `then` to extend a nested tuple of configured stages.
The finalizer is one of:

```rust
.finalize_classifier(estimator) -> SKPipelineClassifierBuilder<F, Stages, E>
.finalize_regressor(estimator)  -> SKPipelineRegressorBuilder<F, Stages, E>
```

The finalizer bounds `E: SKSupervisedFit<F>` and `E::Model: SKClassifierPredictor<F>` or
`SKRegressorPredictor<F>`. `fit` maps over the stage tuple, producing a tuple of fitted
models; the pipeline model stores that tuple and the final model. The type system enforces
the scalar `F` and the final output conversion.

Alternative considered: a runtime `Vec<Box<dyn Stage>>` (simpler but loses the compile-time
guarantee the PRD asks for).

### Decision 4 — Fit and predict

`fit` starts from the borrowed input view, fits stage 1, transforms forward, fits stage 2,
and so on, then fits the final estimator on the last block. `predict` repeats the
transformations with the fitted stages and delegates to the final model. The pipeline
model is immutable and `Send + Sync`.

### Decision 5 — Representation flow

The block preserves dense/sparse. A stage that outputs sparse keeps CSR; a stage that
outputs dense produces dense. The pipeline never densifies on its own. If a stage rejects
the incoming representation, its `UnsupportedRepresentation` error is surfaced.

### Decision 6 — Validation and observability

The builder rejects an empty stage list at finalization. Runtime errors come from the
stages and the final model. The pipeline records a `fit`/`predict` span and a stage span per
stage; candidate estimators record their own backend.

### Decision 7 — File layout

```
crates/sciencekit_pipeline/src/
  lib.rs
  block.rs
  stage.rs
  pipeline/{mod.rs, builder.rs, classifier.rs, regressor.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines. Compile-fail
tests live in `tests/` doctests or a `trybuild`-style harness, not in the library tests.

## Risks / Trade-offs

- **Type-level tuple machinery is complex** → keep the arity bounded and test compile-fail
  cases; the runtime code stays simple.
- **First stage borrows the input, later stages own blocks** → avoids copying the input;
  the boundary is documented.
- **Existing transformer outputs need adapters** → four `From` impls; the impute crate owns
  its two.
- **No FeatureUnion/DAG** → separate changes.
- **Compile errors can be cryptic** → provide a small `SKPipeline::start` helper and
  documented examples.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. FeatureUnion, DAGs and streaming are the next changes.
