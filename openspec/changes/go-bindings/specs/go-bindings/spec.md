# Spec Delta

## Purpose

Defines the Go module: cgo surface, typed wrappers, array conversion, errors, lifecycle and
packaging.

## ADDED Requirements

### Requirement: Module and cgo surface

The Go module SHALL load `sciencekit_ffi` through cgo and expose typed builders/models.

#### Scenario: Build and import

- **WHEN** a Go program imports the module
- **THEN** it builds with cgo and the native library available

### Requirement: Array conversion

Go `[]float64`/`[]float32` matrices SHALL convert to `SKFfiArray` with shape validation and
output copying.

#### Scenario: Dense matrix

- **WHEN** a Go matrix is passed to fit/predict
- **THEN** the native call reads it correctly and returns a Go matrix

#### Scenario: Shape mismatch

- **WHEN** the slice length does not match the shape
- **THEN** the wrapper returns a Go error before the native call

### Requirement: Operations

Builders/models SHALL expose fit, predict, transform, score, save and load.

#### Scenario: Fit and predict

- **WHEN** a builder is configured and fitted
- **THEN** predict returns a Go slice with the expected values

### Requirement: Errors and lifecycle

Native status codes SHALL map to Go errors, and handles SHALL be closed explicitly or by a
runtime finalizer exactly once.

#### Scenario: Native error

- **WHEN** a native call fails
- **THEN** the Go error carries the status and message

#### Scenario: Finalizer

- **WHEN** a wrapper becomes unreachable without Close
- **THEN** the finalizer frees the handle once

### Requirement: Packaging and tests

The module SHALL include `go.mod`, examples, tests and CI instructions.

#### Scenario: Test suite

- **WHEN** `go test ./...` runs
- **THEN** every family's fixture passes
