# Spec Delta

## Purpose

Defines the OpenTelemetry export surface: trace, log and metric pipelines, resource and
transport configuration, sampling, operation bridging and lifecycle.

## ADDED Requirements

### Requirement: Trace export

The telemetry crate SHALL export the library's operation spans to an OTLP collector when
enabled, preserving span names, attributes and error status.

#### Scenario: Fit span exported

- **WHEN** a fit runs with telemetry enabled
- **THEN** the collector receives a `fit` span with rows, columns, backend and execution
  mode

#### Scenario: Error status

- **WHEN** an operation fails
- **THEN** the exported span records the error status and message

### Requirement: Log export

Structured `tracing` events SHALL be exported as OpenTelemetry logs with their fields and
severity.

#### Scenario: Structured event

- **WHEN** the library logs a structured event
- **THEN** the collector receives a log record with the same fields and severity

### Requirement: Metric export

The crate SHALL export operation metrics: duration histogram, operation counter, rows and
columns histograms, and error counter, labelled by operation and backend.

#### Scenario: Duration histogram

- **WHEN** operations complete
- **THEN** the collector receives duration observations labelled by operation/backend

#### Scenario: Error counter

- **WHEN** an operation fails
- **THEN** the error counter increments with the operation label

### Requirement: Configuration and resource

The builder SHALL configure endpoint, transport (gRPC/HTTP), headers, timeout, resource
attributes (service name/version/environment), sampler and batch settings.

#### Scenario: Resource attributes

- **WHEN** a service name/version is configured
- **THEN** every exported signal carries those resource attributes

#### Scenario: Invalid endpoint

- **WHEN** the endpoint is malformed
- **THEN** initialization returns a structured error

### Requirement: No-op default and lifecycle

Without the feature or without initialization, the crate SHALL produce no OTLP traffic; an
initialized pipeline SHALL support `force_flush` and `shutdown` and be safe to drop.

#### Scenario: Default is silent

- **WHEN** telemetry is not initialized
- **THEN** no exporter is created and no network traffic is produced

#### Scenario: Flush before shutdown

- **WHEN** `force_flush` is called
- **THEN** pending signals are handed to the exporter before returning or timing out

#### Scenario: Double initialization

- **WHEN** initialization is called twice
- **THEN** the second call returns a structured error instead of installing a second global
  pipeline

### Requirement: Concurrency

Telemetry initialization, observation and shutdown SHALL be safe under concurrent use, and
concurrent operations SHALL export consistent signals.

#### Scenario: Concurrent operations

- **WHEN** several threads run operations with telemetry enabled
- **THEN** spans/metrics/logs are exported without panic or data race
