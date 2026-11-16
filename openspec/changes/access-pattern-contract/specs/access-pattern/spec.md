# access-pattern Specification

## Purpose

Defines how an estimator declares the data-access pattern it is designed for, how that
declaration maps to the parity memory tiers, which source kinds each pattern may consume,
and the doctrine that keeps out-of-core honest for algorithms whose mathematics cannot
stream.

## ADDED Requirements

### Requirement: Every public estimator declares exactly one access pattern

Every public estimator, transformer and clusterer SHALL expose its access pattern through a
shared declaration contract, readable before fit, immutable for the estimator's lifetime,
and identical for every operation of that estimator.

#### Scenario: Declaration is readable before fit

- **WHEN** a consumer holds a configured estimator that has never been fitted
- **THEN** the access pattern is available without touching data and without performing an
  operation

#### Scenario: One estimator never reports two patterns

- **WHEN** the same configured estimator executes fit and then transform
- **THEN** both operations observe the same declared pattern

### Requirement: Four-tier pattern vocabulary

The access-pattern vocabulary SHALL contain exactly four values with fixed semantics:
`Sequential` (single pass or incremental update; parity tier `streaming`), `RandomAccess`
(positional access by index; parity tier `mappable`), `Iterative` (multi-pass over
reopenable data; parity tier `chunked`) and `FullMemory` (the whole design must be resident;
parity tier `full-memory`).

#### Scenario: Vocabulary maps to parity tiers

- **WHEN** the parity matrix records a tier for an implemented estimator
- **THEN** that tier equals the tier of the estimator's declared pattern

#### Scenario: Unknown value is unrepresentable

- **WHEN** code attempts to construct an access pattern outside the four values
- **THEN** the attempt does not compile, because the vocabulary is a closed enum

### Requirement: Declared pattern constrains the accepted source kind

An operation SHALL reject a source kind incompatible with its declared pattern before
processing any element: `Sequential` accepts sequential lazy sources, `RandomAccess`
accepts mappable sources, `Iterative` accepts reopenable sources of either kind, and
`FullMemory` accepts in-memory views only. Rejection SHALL use a structured error naming
the declared pattern and the rejected source kind.

#### Scenario: Sequential estimator refuses a mappable source

- **WHEN** a `Sequential` estimator receives a mappable source
- **THEN** the operation fails before reading any element with a structured error naming
  both sides of the conflict

#### Scenario: Random-access estimator refuses a sequential-only source

- **WHEN** a `RandomAccess` estimator receives a source that cannot be reopened or accessed
  by position
- **THEN** the operation fails before reading any element

#### Scenario: Iterative estimator accepts both reopenable kinds

- **WHEN** an `Iterative` estimator receives either a reopenable lazy source or a mappable
  source
- **THEN** the operation proceeds and each pass sees the same logical data

### Requirement: Operations derive the pattern from the declaration

Every operation SHALL obtain its access pattern from the estimator's declaration; an
operation SHALL NOT pass a hardcoded pattern value into execution resolution or into
source selection.

#### Scenario: Changing the declaration changes the resolved plan

- **WHEN** an estimator's declaration is changed in code from `RandomAccess` to `Iterative`
- **THEN** every operation of that estimator resolves with the new pattern without editing
  the operation bodies

#### Scenario: Span carries the declared pattern

- **WHEN** an operation runs under observability
- **THEN** its span attributes include the declared access pattern alongside the execution
  mode and backend

### Requirement: Full-memory doctrine

An estimator whose mathematics requires the whole design resident (quadratic or cubic
memory) SHALL declare `FullMemory`, SHALL name a documented approximate alternative, and
SHALL reject a non-in-memory source with the existing `UnsupportedOperation` error whose
reason names that alternative. It SHALL NOT attempt an out-of-core path.

#### Scenario: Full-memory estimator refuses a lazy source with guidance

- **WHEN** a `FullMemory` estimator receives a sequential lazy source
- **THEN** the operation fails with `UnsupportedOperation` naming the approximate
  alternative (for example Nystroem, RandomizedSVD or a mini-batch formulation)
- **AND** no element of the source is consumed

#### Scenario: In-memory input still works

- **WHEN** a `FullMemory` estimator receives an in-memory dense or sparse view that fits the
  budget
- **THEN** the operation proceeds normally

### Requirement: Conformance of declarations, sources and matrix

A repository test SHALL enumerate every public estimator and fail when an estimator lacks a
declaration, when its declaration is incompatible with a source kind it accepts, or when its
declaration disagrees with its parity-matrix tier.

#### Scenario: Missing declaration fails the suite

- **WHEN** a new estimator is added without implementing the declaration contract
- **THEN** the conformance test fails and names the estimator

#### Scenario: Matrix drift fails the suite

- **WHEN** an estimator's declaration changes but its matrix row keeps the old tier
- **THEN** the conformance test fails and prints both tiers

### Requirement: Resolution behavior is unchanged

The contract SHALL NOT change how a plan is resolved: automatic intent SHALL still never
fail for incompatibility, and an explicit incompatible mode SHALL still fail with the
pre-existing execution-mode incompatibility error naming both the mode and the declared
pattern.

#### Scenario: Explicit incompatible mode still fails as before

- **WHEN** a consumer explicitly requests an execution mode incompatible with the declared
  pattern
- **THEN** resolution fails with the pre-existing error, named exactly as today

#### Scenario: Automatic intent with a full-memory estimator

- **WHEN** intent is automatic and the estimator declares `FullMemory`
- **THEN** resolution selects an in-memory mode and does not fail on incompatibility

