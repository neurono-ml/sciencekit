# Spec Delta

## Purpose

Defines the Python binding surface: package layout, NumPy zero-copy boundary, estimator and
pipeline wrappers, error mapping, GIL behavior, packaging and tests.

## ADDED Requirements

### Requirement: Python package layout

The binding SHALL expose a `sciencekit` Python package with family submodules and a
top-level namespace re-exporting the common builders and models.

#### Scenario: Import

- **WHEN** a user imports `sciencekit`
- **THEN** the common builders/models are available and family submodules import cleanly

#### Scenario: Family module

- **WHEN** a user imports `sciencekit.linear_model`
- **THEN** its estimators and builders are available

### Requirement: Zero-copy NumPy input

A feature matrix SHALL be accepted as a NumPy array without copying its buffer when the
dtype and layout match, through a read-only view into `ArrayView2`.

#### Scenario: f64 input

- **WHEN** an `f64` NumPy array is passed to `fit` or `predict`
- **THEN** the Rust view borrows the same buffer and the result matches the Rust API

#### Scenario: f32 input

- **WHEN** an `f32` NumPy array is passed
- **THEN** the scalar is preserved and the model uses `f32`

#### Scenario: Unsupported dtype

- **WHEN** an unsupported dtype is passed
- **THEN** the binding raises a clear Python exception

### Requirement: Estimator coverage

Every estimator family SHALL expose a builder wrapper, a fitted model wrapper and the
operations it supports (`fit`, `predict`, `predict_proba`, `transform`, `fit_predict`,
`score`).

#### Scenario: Classifier

- **WHEN** a classifier is built, fitted and used to predict
- **THEN** labels and probabilities are returned as NumPy arrays

#### Scenario: Transformer

- **WHEN** a transformer is fitted and applied
- **THEN** the transformed matrix is returned as a NumPy array

### Requirement: Pipelines and model selection

Pipelines, FeatureUnion, grid search and halving search SHALL be available as Python
objects with the same composition semantics as Rust.

#### Scenario: Pipeline

- **WHEN** a Python pipeline is built from stages and fitted
- **THEN** predict/transform match the equivalent Rust pipeline

#### Scenario: Grid search

- **WHEN** a Python grid search is fitted
- **THEN** the best model, score and cross-validation results are accessible

### Requirement: Error mapping

`SKError` variants SHALL map to stable Python exceptions with the original message and
context.

#### Scenario: Shape mismatch

- **WHEN** the input shape is invalid
- **THEN** a `ValueError`-style exception is raised with the expected/found shapes

#### Scenario: Not fitted

- **WHEN** a model method is called before fitting
- **THEN** a dedicated not-fitted exception is raised

### Requirement: GIL and concurrency

Long-running compute SHALL release the GIL, and fitted model objects SHALL be safe to use
from multiple Python threads.

#### Scenario: Fit releases the GIL

- **WHEN** a fit runs on a large input
- **THEN** the GIL is released during the compute phase

#### Scenario: Concurrent prediction

- **WHEN** several Python threads predict with the same model
- **THEN** results are correct and no data race occurs

### Requirement: Packaging and type stubs

The package SHALL ship `pyproject.toml`, a maturin build, type stubs for the public
surface and a pytest suite.

#### Scenario: Wheel builds

- **WHEN** the wheel is built
- **THEN** it installs and imports in a clean environment

#### Scenario: Stubs resolve

- **WHEN** a type checker reads the stubs
- **THEN** the documented classes and methods resolve
