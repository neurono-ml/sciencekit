# Spec Delta

## Purpose

Defines low-rank adapters for matrix-parameter models: configuration, attach/fit/predict,
merge/unmerge, persistence and validation.

## ADDED Requirements

### Requirement: Adapter configuration

`SKLoraAdapter` SHALL be constructed through a builder with `rank`, `alpha` and the target
parameter selection, and SHALL expose the scaling `alpha / rank`.

#### Scenario: Scaling

- **WHEN** an adapter is built with rank 8 and alpha 16
- **THEN** its scaling is `2`

#### Scenario: Invalid rank

- **WHEN** rank is zero
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Frozen base

Attaching an adapter SHALL leave the base model immutable; prediction without a fitted
adapter SHALL equal the original base prediction, and fitting SHALL not change base
parameters.

#### Scenario: Base is frozen

- **WHEN** an adapter is fitted
- **THEN** the base coefficients are unchanged

#### Scenario: Zero adapter

- **WHEN** an unfitted/zero adapter predicts
- **THEN** the output equals the base model

### Requirement: Adapter fit and prediction

Adapter fitting SHALL train only the A/B matrices on the supplied data, and prediction
SHALL use `base + scaling · B A` for the selected parameter matrices.

#### Scenario: Fit reduces loss

- **WHEN** an adapter is fitted on data consistent with a shifted relationship
- **THEN** the training loss decreases relative to the frozen base

#### Scenario: Shape compatibility

- **WHEN** the adapter rank/shape is incompatible with the base parameter
- **THEN** fitting returns `ShapeMismatch`

### Requirement: Merge and unmerge

The adapter SHALL merge into a standalone native model and unmerge back to the frozen base,
preserving predictions within tolerance.

#### Scenario: Merge round-trip

- **WHEN** an adapter is merged and unmerged
- **THEN** predictions before/after match within tolerance

### Requirement: Persistence

The adapter SHALL export/import through the model state with metadata (rank/alpha/target)
and A/B tensors.

#### Scenario: Round-trip

- **WHEN** an adapter is exported and imported
- **THEN** its predictions match the original within tolerance

### Requirement: Validation and concurrency

The adapter SHALL reject unsupported base families, non-finite data, invalid rank/alpha and
shape mismatches; fitted adapters SHALL be safely shareable.

#### Scenario: Unsupported base

- **WHEN** an adapter is attached to a tree model
- **THEN** construction returns `UnsupportedOperation`

#### Scenario: Concurrent prediction

- **WHEN** several threads predict with the same fitted adapter
- **THEN** every result equals the sequential result
