# ffi-abi Specification

## Purpose
Defines the stable C ABI: opaque handles, status codes, array descriptors, lifecycle,
versioning, panic containment and thread safety.

## Requirements

### Requirement: Opaque handles

Every builder, fitted model, splitter and workflow object SHALL be represented by an opaque
handle created and freed by the library; bindings SHALL never inspect its layout.

#### Scenario: Create and free

- **WHEN** a binding creates a builder and frees it
- **THEN** no memory is leaked and a null handle is rejected

#### Scenario: Use after free

- **WHEN** a freed handle is passed to an operation
- **THEN** the library returns an invalid-handle status and never dereferences it

### Requirement: Status codes and last error

Every fallible entry point SHALL return an integer status and SHALL expose a thread-local
last-error message for non-success statuses.

#### Scenario: Shape mismatch

- **WHEN** an operation receives incompatible shapes
- **THEN** it returns the shape status and a message with expected/found dimensions

#### Scenario: Success clears error

- **WHEN** an operation succeeds
- **THEN** the last-error value is empty

### Requirement: Array descriptors

Inputs SHALL be passed as borrowed array descriptors (`dtype`, `shape`, `strides`, `data`,
`len`); outputs SHALL be returned as owned descriptors released by the library.

#### Scenario: Borrowed input

- **WHEN** a binding passes a contiguous f64 array descriptor
- **THEN** the library reads it without copying

#### Scenario: Invalid descriptor

- **WHEN** the descriptor length does not match the shape
- **THEN** the library returns an invalid-array status before reading data

### Requirement: Operations

The ABI SHALL expose fit, predict, transform, score, model save/load and builder setters
for the supported estimators.

#### Scenario: Fit and predict

- **WHEN** a binding fits a model and predicts on an array
- **THEN** the output descriptor contains the expected values

#### Scenario: Model persistence

- **WHEN** a model is saved and loaded through the ABI
- **THEN** predictions match the original

### Requirement: Panic containment

No Rust panic SHALL unwind across the ABI; every entry point SHALL catch panics and return
an internal-error status.

#### Scenario: Panic becomes status

- **WHEN** an internal panic occurs
- **THEN** the binding receives an internal-error status and the process stays alive

### Requirement: Versioning and thread safety

The ABI SHALL expose a version/feature query, and handles SHALL be safe to use from
multiple threads according to their documented sharing rules.

#### Scenario: Version query

- **WHEN** a binding loads the library
- **THEN** it can read the ABI version and feature flags

#### Scenario: Concurrent predict

- **WHEN** several threads predict with the same model handle
- **THEN** results are correct and no data race occurs
