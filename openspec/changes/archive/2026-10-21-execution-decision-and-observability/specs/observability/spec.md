## Purpose

Observability for library operations: `tracing` spans and structured logs emitted around
fitting, transforming and predicting, with an opt-in OpenTelemetry export path so consumers
can collect traces without changing application code.

## ADDED Requirements

### Requirement: Operations emit tracing spans
The library SHALL emit a `tracing` span for each public operation (fit, transform, predict)
and log structured fields (shape, execution mode, backend, duration) within that span.

#### Scenario: An operation is traceable
- **WHEN** a consumer performs a fit operation with a subscriber collecting spans
- **THEN** a span for the operation is recorded with its structured fields

#### Scenario: An error is recorded on the span
- **WHEN** an operation fails
- **THEN** the span records the error so it is visible in the trace

### Requirement: OpenTelemetry export is opt-in
The library SHALL provide an opt-in OpenTelemetry export path (`tracing-opentelemetry`) gated
behind a feature flag, so the default build emits no OTLP traffic.

#### Scenario: Export is disabled by default
- **WHEN** the crate is built without the observability-export feature
- **THEN** no OpenTelemetry exporter is initialized and no OTLP traffic is produced

#### Scenario: Export activates on the feature
- **WHEN** the observability-export feature is enabled and a consumer initializes the exporter
- **THEN** spans are exported to the configured OpenTelemetry collector