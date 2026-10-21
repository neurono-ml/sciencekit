## Purpose

Guarantees that scalar type mismatches between pipeline stages fail at compile time instead of forcing runtime casts.

## ADDED Requirements

### Requirement: Pipeline threads a single scalar across stages

The system SHALL parameterize `SKPipeline` over `F: SKFloat` so a transformer output in `F` only feeds estimators accepting `SKDataView<F>`.

#### Scenario: Homogeneous f32 pipeline compiles and runs

- **WHEN** a pipeline chains `f32` transformer and regressor stages
- **THEN** `fit`/`predict` succeed end to end with `Array2<f32>` between stages and `Array1<f32>` predictions.

#### Scenario: Mixed-scalar chaining fails at compile time

- **WHEN** a user attempts to chain an `f32` stage into an `f64` stage
- **THEN** compilation fails; an explicit cast stage is required to bridge scalars.

### Requirement: Scorers follow the prediction scalar

The system SHALL provide continuous scorers over `ArrayView1<F>` and label scorers over `ArrayView1<i64>`, matching the regressor/classifier split.

#### Scenario: f32 regression scoring without cast

- **WHEN** scoring `f32` predictions against `f32` truth
- **THEN** the metric is computed directly with no `f64` conversion in the observable API.
