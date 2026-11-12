# Proposal

## Why

The PRD promises compile-time validation of pipeline step compatibility. Today a user
chains transformers manually, and a feature-count or scalar mismatch only surfaces at
runtime. A typed linear pipeline makes incompatible chains unrepresentable and lets one
`fit`/`predict` call run the whole chain.

## What Changes

- New sub-crate `sciencekit_pipeline` with a linear, compile-time typed `SKPipeline`.
- A stage builder (`then`) whose type records the accumulated stages, plus final
  classifier/regressor finalizers that type the prediction surface.
- An internal `SKFeatureBlock<F>` medium so dense and sparse outputs flow between stages
  without losing representation; existing transformer outputs are adapted through `From`
  conversions.
- `fit` fits every transformer on the training block, transforms it forward and fits the
  final estimator; `predict_labels`/`predict_probabilities`/`predict` transform new data and
  delegate to the final model.
- Representation parity: dense and sparse blocks flow through the chain; no densification.
- Validation: a stage whose output cannot convert into the next input, feature-count
  mismatch at runtime (for adapters), empty pipeline and final-stage errors.
- Model accessors: fitted stages, final model, stage count and the pipeline's feature
  count.
- Explicit non-goals: parallel feature unions (W6.7), DAGs (W6.8), streaming (W6.9),
  `memmap2` and export.
- Companion tests: compile-fail doctests for incompatible chains, hand-checkable
  fit/predict, dense/sparse parity, nested pipelines, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `typed-pipeline`: the linear pipeline builder, stage typing, fitted pipeline models,
  classifier/regressor finalization, representation flow and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_pipeline` workspace member; root `Cargo.toml`
  membership; `From` adapters for existing transformer outputs.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** unblocks FeatureUnion (W6.7), DAGs (W6.8) and the streaming pipeline
  (W6.9).
- **Acceptance criteria:** small and large data, concurrent prediction, companion
  `*_tests.rs` modules and compile-fail tests. Export is deferred to Wave 7.
