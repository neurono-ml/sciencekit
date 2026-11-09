# Design

## Context

See `proposal.md` — Why. `sciencekit_common` emits `tracing` spans with
`SKOperationAttributes`; its `opentelemetry_layer` is a minimal trace bridge. The workspace
already pins `opentelemetry 0.32` and `tracing-opentelemetry 0.33`. The telemetry crate
owns the SDK, exporters, resource and lifecycle so common stays light.

## Goals / Non-Goals

**Goals:**
- Traces, logs and metrics in one configured pipeline.
- Resource attributes and no-op default.
- Safe flush/shutdown.

**Non-Goals:**
- Vendor-specific backends, tracing UI, profiling, GPU telemetry.
- Changing the span field contract.

## Decisions

### Decision 1 — `sciencekit_telemetry` crate

The crate builds the OTel SDK pipelines and bridges `tracing` spans/events. Library crates
keep emitting `tracing`; they do not depend on the telemetry crate. Operators call
`SKTelemetry::init(config)` once at startup.

### Decision 2 — Configuration builder

```rust
SKTelemetryConfig::builder()
    .service_name("sciencekit")
    .service_version(env!("CARGO_PKG_VERSION"))
    .endpoint("http://localhost:4317")
    .transport(SKTelemetryTransport::Grpc)
    .sampler(SKTelemetrySampler::ParentBased(0.1))
    .batch(...)
    .build()?
```

Resource attributes are attached to traces, logs and metrics. Invalid endpoint/transport/
sampler returns `InvalidHyperparameter`.

### Decision 3 — Metrics bridge

A `tracing` layer observes span close/error and records:
- `sciencekit.operation.duration` histogram (ms) labelled by operation/backend/mode;
- `sciencekit.operation.count` counter;
- `sciencekit.operation.rows`/`columns` histograms;
- `sciencekit.operation.errors` counter.
Metric names/units are documented; cardinality is bounded by the closed operation/backend
enums.

### Decision 4 — Logs

`tracing` events are exported as OTel logs with severity, target, message and fields. The
existing span fields are preserved.

### Decision 5 — Lifecycle

`init` installs a global tracer/meter/logger provider and a subscriber; `force_flush`
flushes all providers with a timeout; `shutdown` flushes then shuts down. A second `init`
returns `InvalidHyperparameter` (or `UnsupportedOperation`). Drop attempts a best-effort
flush.

### Decision 6 — Features

`otlp-grpc` and `otlp-http` select the exporter transport; `telemetry` enables the crate.
All off by default. No dependency in `sciencekit_common`.

### Decision 7 — File layout

```
crates/sciencekit_telemetry/src/
  lib.rs
  config/{mod.rs, builder.rs, resource.rs, sampler.rs, *_tests.rs}
  pipeline/{mod.rs, traces.rs, logs.rs, metrics.rs, layer.rs, *_tests.rs}
  lifecycle/{mod.rs, init.rs, shutdown.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Global provider initialization is process-wide** → explicit lifecycle and double-init
  error.
- **Cardinality explosion** → bounded labels from closed enums; no user IDs in metrics.
- **Exporter network failures** → batch processor drops with logs and never blocks
  operations.
- **Version alignment** → pin opentelemetry 0.32 family; spike if tracing-opentelemetry
  requires a different minor.
- **Test flakiness** → in-memory exporter and deterministic flush.

## Migration Plan

Additive: new crate and feature. No existing behavior changes; default build stays silent.

## Open Questions

None blocking. A Prometheus exporter is a future feature.
