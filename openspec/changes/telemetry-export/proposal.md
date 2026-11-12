# Proposal

## Why

The observability capability already emits `tracing` spans and has an opt-in
`tracing-opentelemetry` layer, but it exports traces only, has no OTLP collector
configuration, no resource attributes, no metrics or log bridge and no lifecycle
(shutdown/flush). Production users need all three signals with a no-op default.

## What Changes

- New `sciencekit_telemetry` crate with OTLP trace, log and metric pipelines over
  `opentelemetry 0.32`/`opentelemetry_sdk 0.32`/`opentelemetry-otlp 0.32` and
  `tracing-opentelemetry 0.33`.
- A configuration builder: endpoint, transport (gRPC/HTTP), headers, timeout, resource
  attributes (service name/version/environment), sampling and batch settings.
- Trace bridge from the existing `tracing` spans; log bridge for structured events; metric
  bridge for operation duration/count, rows/columns, backend and errors.
- A no-op default: no exporter is initialized and no OTLP traffic is produced.
- Explicit lifecycle: `init`, `shutdown`, `force_flush` and drop-safe cleanup.
- Validation: invalid endpoint/transport, missing service name, exporter initialization
  failure and double initialization.
- Explicit non-goals: vendor-specific backends, tracing UI, profiling and GPU telemetry.
- Companion tests: in-memory exporter capture of spans/logs/metrics, no-op default,
  attribute mapping, flush/shutdown and concurrent operations.

## Capabilities

### New Capabilities

- `telemetry-export`: OTLP trace/log/metric pipelines, resource/configuration, sampling,
  bridge from library operations and lifecycle.

### Modified Capabilities

- `observability`: the opt-in export path now covers traces, logs and metrics through the
  telemetry crate.

## Impact

- **Code:** new `crates/sciencekit_telemetry` workspace member; `sciencekit_common`
  observability docs/feature wiring.
- **APIs:** additive; no-op by default.
- **Dependencies:** `opentelemetry 0.32`, `opentelemetry_sdk 0.32`,
  `opentelemetry-otlp 0.32`, `tracing-opentelemetry 0.33` (MSRV 1.75–1.85).
- **Downstream:** lets operators collect traces, logs and metrics without changing library
  code.
- **Acceptance criteria:** all three signals captured, resource attributes correct, no-op
  default, lifecycle safe and concurrent operations observed.
