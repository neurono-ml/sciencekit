# mappable-fit Specification

## Purpose

Makes memory-mapped, randomly accessible datasets first-class training inputs for the
estimators whose declared pattern is `RandomAccess`, without densification and without
changing in-memory results.

## ADDED Requirements

### Requirement: Random-access estimators fit from a mappable source

Every estimator whose declared access pattern is `RandomAccess` SHALL expose a fit that
accepts an `SKMappableSource` and SHALL reject a source that cannot provide positional
access before reading any row.

#### Scenario: KNN fits from a mapped design

- **WHEN** `SKKNeighborsClassifier` is fitted from an `SKMappableSource` backed by a mapped
  file
- **THEN** the fit completes without materializing the whole design
- **AND** predictions equal the in-memory fit within the documented tolerance

#### Scenario: Non-mappable source is rejected early

- **WHEN** a `RandomAccess` estimator receives a sequential-only source
- **THEN** the fit fails with a structured error naming the required source kind and no row
  is read

### Requirement: No densification and a memory guard

The mappable path SHALL never densify a sparse design, SHALL account its reads against the
execution plan's memory guard, and SHALL reuse the existing dense/CSR dispatch.

#### Scenario: CSR mapping stays sparse

- **WHEN** the mapped source exposes CSR layout
- **THEN** the fit reads stored entries only and the sparse result equals the in-memory
  sparse fit

### Requirement: Model ownership over mapped storage is explicit

Each estimator SHALL document whether its model holds a shared handle to the mapped source
or owns its compact state; export SHALL materialize owned tensors regardless, so persistence
and import are unaffected.

#### Scenario: KNN model outlives the source handle

- **WHEN** a fitted KNN model holding a shared mapping is used after the caller drops its
  reference to the source
- **THEN** predictions still work, because the handle keeps the mapping alive

#### Scenario: Exported model owns its state

- **WHEN** a model fitted from a mapped source is exported
- **THEN** the exported state is fully owned and imports without the original file

### Requirement: Parity and unchanged in-memory path

The mappable fit SHALL produce the same model as the in-memory fit of the same logical
data, and the existing in-memory fit SHALL keep its behavior, tests and signatures.

#### Scenario: Dense and mapped fits agree

- **WHEN** the same dataset is fitted in memory and from a mapping
- **THEN** the two models predict identically within the documented tolerance

#### Scenario: Existing callers compile unchanged

- **WHEN** an existing caller fits from an in-memory view
- **THEN** the call compiles and behaves exactly as before
