# Spec Delta

## Purpose

Defines the Julia package: ccall surface, array conversion, builders/models, errors, library
discovery and tests.

## ADDED Requirements

### Requirement: Package loading

The Julia package SHALL load the native library through a documented discovery mechanism
and expose the module.

#### Scenario: Using the package

- **WHEN** Julia runs `using ScienceKit`
- **THEN** the module loads and exports the documented builders/models

### Requirement: Array conversion

Julia `Array{Float32/Float64}` SHALL convert to/from `SKFfiArray` with shape validation and
`GC.@preserve` around native calls.

#### Scenario: Dense matrix

- **WHEN** a Julia matrix is passed to fit/predict
- **THEN** the native call reads it without corruption and the output is a Julia matrix

#### Scenario: Non-contiguous array

- **WHEN** a non-contiguous view is passed
- **THEN** the wrapper copies to a contiguous array or raises a clear error

### Requirement: Operations

Builders/models SHALL expose fit, predict, transform, score, save and load with Julia
keyword arguments.

#### Scenario: Fit/predict

- **WHEN** a builder is configured and fitted
- **THEN** predict returns a Julia vector/matrix with the expected values

### Requirement: Errors and lifecycle

Status codes SHALL map to Julia exceptions with messages, and handles SHALL be finalized
when wrappers are collected.

#### Scenario: Invalid input

- **WHEN** a native call fails
- **THEN** Julia raises an exception with the status/message

#### Scenario: Finalizer

- **WHEN** a model wrapper becomes unreachable
- **THEN** its native handle is freed exactly once

### Requirement: Packaging and tests

The package SHALL include `Project.toml`, `src/`, `test/` and CI instructions.

#### Scenario: Test suite

- **WHEN** `Pkg.test()` runs
- **THEN** every family's fixture passes
