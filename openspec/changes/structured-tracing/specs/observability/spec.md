# observability Specification Delta

## MODIFIED Requirements

### Requirement: Operations emit tracing spans
The library SHALL emit a `tracing` span for each public operation (fit, transform,
predict, partial_fit, fit_streaming and score), logging structured fields (shape,
execution mode, backend, declared access pattern and duration) within that span, and the
span SHALL nest under the ambient caller span so a pipeline, a search or a streamed fit
produces a single trace tree.

#### Scenario: An operation is traceable
- **WHEN** a consumer performs a fit operation with a subscriber collecting spans
- **THEN** a span for the operation is recorded with its structured fields

#### Scenario: An error is recorded on the span
- **WHEN** an operation fails
- **THEN** the span records the error so it is visible in the trace

#### Scenario: An operation nests under its caller
- **WHEN** an estimator operation runs inside a pipeline stage or a search fold
- **THEN** its span is a child of the caller's span and shares the trace identifier

#### Scenario: Every inference emits a trace
- **WHEN** a model predicts, transforms or scores
- **THEN** the inference emits a span with the same vocabulary as training

## ADDED Requirements

### Requirement: Pipeline operations nest stage spans

A pipeline operation SHALL emit one parent span `sciencekit.pipeline` for the pipeline and
one child span `sciencekit.pipeline.stage` per stage, in both fit and predict, so a single
call produces one trace tree ordered by the stage order. The legacy span name
`pipeline_graph_node` SHALL NOT be emitted anymore.

#### Scenario: A fit nests stages in order

- **WHEN** a pipeline fits a chain of three stages
- **THEN** the trace contains one pipeline span and three ordered stage spans, each nesting
  its estimator operation

#### Scenario: A predict nests stages identically

- **WHEN** the fitted pipeline predicts
- **THEN** the trace repeats the same hierarchy with the predict operation

### Requirement: Iterative and streamed operations span epochs and batches

An iterative or streamed operation SHALL emit one `sciencekit.operation.epoch` span per
epoch at DEBUG level and `sciencekit.operation.batch` spans at TRACE level nested under
the operation span, carrying the epoch or batch index, so a full execution cycle is
inspectable end to end.

#### Scenario: A streamed fit reports every batch

- **WHEN** a `fit_streaming` processes five batches in two epochs
- **THEN** the trace contains two epoch spans at DEBUG and five batch spans at TRACE with
  ordered indices, all nested under the operation span

#### Scenario: Prefetch stays inside the trace

- **WHEN** the streaming driver produces batches on its producer thread
- **THEN** the producer's spans are children of the operation span, not orphan roots

### Requirement: Batch spans are capped per epoch

Only the first `SK_TRACE_BATCH_SPANS_PER_EPOCH` (32) batches of each epoch SHALL emit
individual spans; the remaining batches SHALL be aggregated into one `batches_aggregated`
span event on the epoch span, carrying the batch count, total rows, total bytes and total
duration, so span volume stays bounded.

#### Scenario: A long epoch aggregates its tail

- **WHEN** an epoch processes 40 batches
- **THEN** the trace contains 32 batch spans with ordered indices and one
  `batches_aggregated` event on the epoch span reporting 8 batches with their total rows,
  bytes and duration

#### Scenario: Early stop leaves a complete trace

- **WHEN** an iterative fit stops early
- **THEN** the trace contains the epochs actually run and the operation span records the
  early-stop outcome as a span event

### Requirement: Error status marks the whole chain

The span where an error is detected SHALL record the error message and kind, and every
ancestor span SHALL record status ERROR, so a trace shows the complete failure path.

#### Scenario: A failed batch marks epoch, operation, stage and pipeline

- **WHEN** a batch fails inside a streamed pipeline fit
- **THEN** the batch span records the message and kind and every ancestor span carries the
  ERROR status with `child failed`

### Requirement: Trace context crosses the FFI

The FFI SHALL expose the current W3C `traceparent` and SHALL accept an optional
`traceparent` on every entry point, nesting the operation under the remote parent, so a
Go, Lua or Python caller continues its own trace.

#### Scenario: A binding continues its caller's trace

- **WHEN** a binding calls into the library passing its traceparent
- **THEN** the emitted spans share the caller's trace identifier

#### Scenario: A binding reads the current traceparent

- **WHEN** a binding asks for the traceparent while a library operation runs
- **THEN** it receives a W3C-formatted string or an empty string when no trace is active

### Requirement: Span vocabulary is fixed

The span names, the batch-cap event, the step values, the levels and the field names SHALL
form a closed vocabulary enumerated in the design; additions enter only through a spec
change, never ad hoc.

#### Scenario: An unknown span kind fails review

- **WHEN** an estimator introduces a span name, field or level outside the vocabulary
- **THEN** the conformance test that enumerates the vocabulary fails and names it
