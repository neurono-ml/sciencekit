# Spec Delta

## Purpose

Defines the model-side cluster assignment contract: how a fitted clusterer exposes
assignments, how noise is represented and what shape/scalar guarantees consumers rely on.

## ADDED Requirements

### Requirement: Cluster assignment prediction

A fitted clusterer SHALL expose cluster assignments as canonical `i64` indices, one per
input row, independent of the feature scalar, with `-1` reserved for noise.

#### Scenario: Assignments are canonical integers

- **WHEN** a fitted KMeans model predicts over `f32` and `f64` features of identical values
- **THEN** both return identical `Array1<i64>` assignments

#### Scenario: Noise uses the reserved label

- **WHEN** a density clusterer marks a point as noise
- **THEN** its assignment is `-1` and never a fabricated cluster index

#### Scenario: Assignment count follows the input

- **WHEN** a batch of `m` rows is assigned
- **THEN** the result has length `m`

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
