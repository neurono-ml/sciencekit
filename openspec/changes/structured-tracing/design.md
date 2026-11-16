# Design

## Context

`sk_run_operation` emits one span (`sciencekit.operation`) per public operation, the
pipeline graph emits ad-hoc `pipeline_graph_node` spans, the streaming driver produces on
a scoped producer thread, and no rayon section propagates a span. The metric histograms
(`sciencekit.operation.duration/count/rows/columns/errors`) are derived from the close of
the operation span. Three hard facts drive this design: the published span and metric
names are already consumed by `telemetry-export` (renaming breaks consumers), nesting in
`tracing` is thread-local (scoped threads and rayon workers start without a parent), and
batch-level spans are high-volume (levels, not caps, are the control mechanism).

## Decisions

1. **Batch spans at TRACE**: epochs and steps stay at DEBUG, and only the batch spans
   drop to TRACE, so the default INFO build shows pipeline/stage/operation, DEBUG reveals
   the cycle, and TRACE reveals the batches.
2. **Capped batch spans with an epoch summary** (option B): each epoch emits its first
   `SK_TRACE_BATCH_SPANS_PER_EPOCH` (32) batch spans as individual TRACE spans and
   aggregates the rest into one epoch-closing `batches_aggregated` span event carrying
   the batch count, total rows, total bytes and total duration. The cap bounds span
   volume without losing the per-epoch totals; a deterministic 1-in-N downsampling is a
   compatible future evolution if the aggregate is ever insufficient.
3. **Error status on the whole chain**: the origin span records the message and the kind,
   every ancestor span records status ERROR with `child failed`, so a trace shows the
   complete failure path.
4. **Trace context crosses the FFI**: bindings receive and return a W3C `traceparent`, so
   Go/Lua/Python callers continue the same trace.
5. **Breaking rename** (pre-1.0): `pipeline_graph_node` becomes
   `sciencekit.pipeline.stage`; the change is recorded in the changelog.

## Goals / Non-Goals

**Goals**
- One closed span vocabulary with fixed names, fields and levels.
- Nesting by construction, including across threads and parallel sections.
- One operation-level span feeding the metrics layer; nothing below it feeds metrics.

**Non-Goals**
- Changing OTLP export, sampling or log pipelines (`telemetry-export` already shipped).
- Per-row spans (batch is the smallest unit).
- Downsampling knobs (the per-epoch cap of Decision 2 is fixed; 1-in-N downsampling is a
  documented future evolution).

## Span tree

```
sciencekit.pipeline            INFO   operation, stage_count, rows, columns
└─ sciencekit.pipeline.stage   INFO   index, stage_kind, estimator, operation
   └─ sciencekit.operation     INFO   operation, rows, columns, mode, backend, pattern,
      │                              duration_ms, error   ← only level feeding metrics
      ├─ sciencekit.operation.step   DEBUG   step, [index]
      ├─ sciencekit.operation.epoch  DEBUG   index, rows
      │  ├─ sciencekit.operation.batch  TRACE   index, rows, bytes   (first 32 per epoch)
      │  └─ event: batches_aggregated           count, rows, bytes, duration_ms
      └─ sciencekit.operation.step   DEBUG   ...
```

Rules:

- Without a pipeline, `sciencekit.operation` is the root; levels 0 and 1 exist only inside
  a pipeline.
- Every inference (`predict`, `transform`, `score`) emits a root span when called outside
  a pipeline and a stage child inside one — same vocabulary as training.
- Steps use a closed enum: `initialize`, `iterate`, `converge`, `finalize`, `export`.

## Propagation

Nesting is thread-local in `tracing`; the hierarchy requires explicit propagation:

- The streaming driver captures `Span::current()` before spawning the scoped producer
  thread and re-enters it inside the producer.
- Rayon sections capture the ambient span and re-enter it in each closure
  (`span.in_scope(|| ...)`).
- Futures use `.instrument(span)`.
- The conformance test runs a streamed fit with prefetch and asserts one trace tree, not
  an orphan producer span.

## Errors

The span where an error is detected records `error` (message) and `error.kind` (the
`SKError` taxonomy variant). Each ancestor span records status ERROR with
`error = "child failed"`, set on drop, so the whole path is visible. Point-in-time facts
(early stop, fallback reason) are span events, not fields.

## FFI trace context

The FFI gains two functions:

- `sk_ffi_traceparent()` returns the current W3C `traceparent` string for the ambient
  span (or an empty string when no trace is active), so a binding can hand it to its own
  instrumentation.
- Every FFI entry point accepts an optional `traceparent`; when present, the operation
  span nests under that remote parent, so a Go/Lua/Python caller continues its own trace.

The FFI keeps its own ABI surface; the context is a plain string parameter and a plain
string return, so no binding needs OpenTelemetry types.

## Compatibility

- `sciencekit.operation` and the metric names stay unchanged.
- `pipeline_graph_node` is renamed to `sciencekit.pipeline.stage`; the rename is a
  telemetry-visible breaking change recorded in the changelog, acceptable pre-1.0.
- Fields `pattern` and `bytes` are additive.

## Conformance

A workspace test enumerates the span vocabulary and fails on an unknown name or field, and
a hierarchy test asserts the full tree for a pipeline fit, a pipeline predict, a streamed
fit with prefetch (cross-thread), and an early-stopped fit.

## Risks

- **Volume at TRACE:** bounded by the per-epoch cap (Decision 2); the compatible
  downsampling evolution is documented here so it cannot surprise.
- **Propagation leaks:** the cross-thread conformance test is the guard; a span created
  outside the vocabulary fails the suite.
- **FFI ABI:** the traceparent is an optional string parameter; bindings that pass an
  empty string get today's behavior.
