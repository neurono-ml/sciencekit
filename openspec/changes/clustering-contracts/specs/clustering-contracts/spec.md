# Spec Delta

## Purpose

Defines the model-side cluster assignment contract: how a fitted clusterer exposes
assignments, how noise is represented without a numeric sentinel, and what shape, scalar
and interoperability guarantees consumers rely on.

## ADDED Requirements

### Requirement: Typed cluster assignments

A fitted clusterer SHALL expose one assignment per input row as a typed value that is
either a cluster identifier or the explicit noise variant, independent of the feature
scalar.

#### Scenario: KMeans returns cluster variants

- **WHEN** a fitted KMeans model assigns a batch
- **THEN** every element is a cluster variant and no element is noise

#### Scenario: DBSCAN returns the noise variant

- **WHEN** a density clusterer marks a point as noise
- **THEN** its assignment is the explicit `Noise` variant, never a fabricated cluster
  identifier

#### Scenario: Assignment count follows the input

- **WHEN** a batch of `m` rows is assigned
- **THEN** the result has length `m`

### Requirement: Compact assignment representation

The assignment type SHALL use a niche-based cluster identifier so that the noise variant
and a cluster identifier fit in a 32-bit-sized value, and SHALL NOT rely on a plain signed
integer sentinel.

#### Scenario: Size is bounded

- **WHEN** the assignment type is inspected
- **THEN** its size is at most four bytes per element

#### Scenario: No sentinel collision

- **WHEN** a cluster identifier is constructed
- **THEN** it cannot collide with the noise representation

### Requirement: Interoperability conversion

The library SHALL provide explicit conversions between the typed assignment and the
`i64` convention used by external tools, mapping `Noise` to `-1` and a cluster identifier
to its non-negative index.

#### Scenario: Noise maps to -1

- **WHEN** a `Noise` assignment is converted for external interop
- **THEN** it becomes `-1`

#### Scenario: Cluster maps to its index

- **WHEN** a cluster assignment is converted for external interop
- **THEN** it becomes its non-negative cluster index

#### Scenario: Invalid external value is rejected

- **WHEN** an external `i64` below `-1` is converted into the typed assignment
- **THEN** the conversion returns a structured error instead of inventing a cluster

#### Scenario: Out-of-range external value is rejected

- **WHEN** an external `i64` above the maximum representable cluster index is converted
- **THEN** the conversion returns a structured error and never panics

### Requirement: Assignment input boundary

The assignment method SHALL accept the same fallible `SKDataView` conversion seam as the
other predictors and SHALL validate the feature count against the fitted model.

#### Scenario: Dense and sparse inputs share the boundary

- **WHEN** a fitted clusterer receives a dense or CSR view with the fitted feature count
- **THEN** the assignment succeeds through the same method

#### Scenario: Feature-count mismatch is rejected

- **WHEN** a batch has a different feature count from the fitted model
- **THEN** the method returns `ShapeMismatch` before producing assignments

### Requirement: Assignment models are shareable

Fitted clusterers SHALL be safely shareable across threads and SHALL produce the same
assignments under concurrent calls as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads assign with the same fitted model
- **THEN** every call completes without locking and returns its sequential result
