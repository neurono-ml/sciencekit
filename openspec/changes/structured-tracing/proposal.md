# Proposal

## Why

The observability contract today guarantees one span per public operation with its
structured fields, but nothing about hierarchy: a pipeline fit emits disconnected spans,
an iterative fit reports a single block with no per-epoch detail, a streamed fit is
invisible batch by batch, and inference shares no vocabulary with training. With the
parity program adding ~150 estimators, this must be fixed in the contract now, otherwise
each new algorithm invents its own tracing shape and the traces cannot be nested or
aggregated.

## What Changes

- Define the span hierarchy: a pipeline operation is a parent span, each stage is a child
  span, each estimator operation (`fit`, `transform`, `predict`, `partial_fit`,
  `fit_streaming`, `score`) is a stage child, each algorithm step (initialization,
  iteration, convergence check, export) is a nested span, and each epoch and each batch is
  a nested span inside a streamed operation.
- Require every inference to emit its own trace with the same vocabulary as training
  (`sk.pipeline`, `sk.pipeline.stage`, `sk.operation`, `sk.operation.step`,
  `sk.operation.epoch`, `sk.operation.batch`), including the access pattern, the backend,
  the resolved plan and the batch/epoch index.
- Require `SKOperationObservation` to nest under the ambient span, so callers that run
  inside a pipeline or a search produce a single trace tree.
- Add the conformance test: a pipeline over a mock estimator emits exactly one pipeline
  span, one span per stage, and the estimator's step spans, in both fit and predict, and a
  streamed fit emits epoch and batch spans with ordered indices.
- Explicit non-goals: metrics/logging export changes (telemetry-export already shipped),
  sampling-rate tuning, and OpenTelemetry collector configuration.

No result changes; the change is purely additive to the observability surface.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `observability`: the single-span-per-operation requirement becomes the span-hierarchy
  requirement, covering nesting, the inference trace, the streaming epochs and batches,
  and the pipeline stage spans.

## Impact

- `sciencekit_common`: span helpers and the nesting convention.
- `sciencekit_pipeline` and the streaming driver: parent/stage/epoch/batch spans.
- Every estimator crate: step spans where an algorithm exposes phases; no behavior change.
- The DAG gains the change in the `contracts` lane after `access-pattern-contract`.
