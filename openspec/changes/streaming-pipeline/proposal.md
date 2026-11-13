# Proposal

## Why

The typed pipeline is batch-only. The PRD's out-of-core differentiator requires processing
batches through the preprocessing stages without materializing the whole dataset. A
streaming pipeline composes the existing `SKLazySource`/`SKSourceProvider` executor with
streaming transformers and a final estimator that can consume batches.

## What Changes

- New `SKStreamingPipeline` in `sciencekit_pipeline`, driven by the shared prefetch
  executor.
- `SKStreamingTransformer<F>` contract: `fit_streaming` over a source provider and
  `transform_batch(SKFeatureBlock<F>) -> SKFeatureBlock<F>`.
- Chain of streaming stages, each fitted by a pass over the source, then applied per batch
  in order.
- Final stage options: a batch estimator fitted on the accumulated transformed blocks, or
  a streaming estimator whose `partial_fit` consumes each transformed batch. The first
  version supports the batch finalizer with an explicit memory budget; a true
  out-of-core estimator finalizer is a later change.
- Multiple epochs for reopenable providers; one pass for one-shot providers.
- Dense and sparse blocks flow through the chain; no densification.
- Validation: empty pipeline, a stage that cannot consume the incoming representation,
  source read errors, feature-count mismatch and an exhausted one-shot provider on a
  multi-epoch request.
- Explicit non-goals: `memmap2`-backed batches (Wave 7), streaming DAGs, GPU, export.
- Companion tests: hand-checkable batch chain, streaming equals batch transform, multiple
  epochs, dense/sparse, source error propagation, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `streaming-pipeline`: the streaming transformer contract, chained batch transformation,
  streaming fitting, epochs and the finalizer policy.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_pipeline` gains the streaming modules.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common` (streaming sources and executor), `ndarray`,
  `sprs`, `rayon`.
- **Downstream:** gives preprocessing an out-of-core path and prepares the memory-mapped
  storage work in Wave 7.
- **Acceptance criteria:** small and large streams, concurrency, companion `*_tests.rs`
  modules. `memmap2` backing and export are deferred to Wave 7.
