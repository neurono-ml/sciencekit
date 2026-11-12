# Spec Delta

## Purpose

Defines the format-agnostic model export/import contract: model metadata, named tensors,
hyperparameters, dtype/shape validation and the export/import traits.

## ADDED Requirements

### Requirement: Export state

The library SHALL expose an export method on fitted models returning a model state that
contains the estimator metadata, a map of named tensors and typed hyperparameters.

#### Scenario: State has metadata and tensors

- **WHEN** a fitted model is exported
- **THEN** the state names the estimator, its scalar dtype and feature count, and exposes
  every learned tensor by a stable name

#### Scenario: Tensor names are unique

- **WHEN** a state contains two tensors with the same name
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Tensor representation

A tensor SHALL carry a stable name, a dtype (`F32`, `F64`, `I64` or `Bool`), a shape and
values whose length matches the shape product.

#### Scenario: Shape and values agree

- **WHEN** a tensor is constructed with a shape whose product differs from the value count
- **THEN** construction returns `ShapeMismatch`

#### Scenario: Dtype matches the model scalar

- **WHEN** an `f64` model exports a floating tensor
- **THEN** the tensor dtype is `F64`

### Requirement: Metadata and hyperparameters

The state SHALL expose estimator name, format version, scalar dtype, feature count, class
labels when applicable and a hyperparameter map of typed values.

#### Scenario: Class labels are preserved

- **WHEN** a classifier is exported
- **THEN** its canonical class table is present in the metadata

#### Scenario: Hyperparameters are typed

- **WHEN** a model has integer, float, boolean, string or list hyperparameters
- **THEN** each is represented by the matching typed value

### Requirement: Import contract

The library SHALL expose an import method that reconstructs a model from a state, and SHALL
validate the estimator name, format version and feature count before accepting it.

#### Scenario: Round-trip

- **WHEN** a fitted model is exported and imported
- **THEN** the imported model produces the same predictions within tolerance

#### Scenario: Wrong estimator is rejected

- **WHEN** a state names a different estimator
- **THEN** import returns `UnsupportedRepresentation`

#### Scenario: Wrong feature count is rejected

- **WHEN** the state's feature count differs from the model's expected count
- **THEN** import returns `ShapeMismatch`

### Requirement: Read-only concurrency

Exported states SHALL be immutable and safely shareable, and concurrent export of the same
model SHALL return identical states.

#### Scenario: Concurrent export agrees

- **WHEN** several threads export the same fitted model
- **THEN** every state equals the single-threaded state
