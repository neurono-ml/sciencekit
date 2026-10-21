## Purpose

Defines the generic scalar contract for predictions, targets, and scorers so f32 and f64 flow end to end without implicit casts.

## ADDED Requirements

### Requirement: Regressor predictions preserve the feature scalar

The system SHALL return regressor predictions in the same scalar type as the input features, with no implicit float cast.

#### Scenario: f32 in, f32 out

- **WHEN** a regressor model fitted on `f32` features predicts over `SKDataView<f32>`
- **THEN** the result is `Array1<f32>` with one element per row and no `f64` intermediate in the observable API.

#### Scenario: f64 in, f64 out

- **WHEN** a regressor model fitted on `f64` features predicts over `SKDataView<f64>`
- **THEN** the result is `Array1<f64>` with one element per row.

#### Scenario: Scalar mismatch is rejected

- **WHEN** `predict` receives features whose scalar differs from the fitted model scalar
- **THEN** the system returns a conversion/shape error before computing any element.

### Requirement: Classifier predictions are dtype-independent labels

The system SHALL return classifier predictions as canonical `Array1<i64>` label indices regardless of feature scalar, and class probabilities in the feature scalar.

#### Scenario: Labels do not depend on float width

- **WHEN** a classifier predicts over `f32` vs `f64` features of identical values
- **THEN** both return identical `Array1<i64>` label vectors.

#### Scenario: Probabilities preserve the feature scalar

- **WHEN** `predict_proba` is called on `f32` features
- **THEN** the result is `Array2<f32>` with rows summing to one within float tolerance.

### Requirement: Targets are generic over the feature scalar

The system SHALL accept continuous targets in the model scalar `F` via `SKTargetView::Continuous(ArrayView1<F>)`, keeping `Integer` and `Nominal` variants unchanged.

#### Scenario: f32 regression targets

- **WHEN** fitting a regressor with `Continuous` `f32` targets
- **THEN** fitting succeeds without an `f64` copy in the observable API.

#### Scenario: Nominal targets still canonicalize

- **WHEN** fitting a classifier with `Nominal` string targets
- **THEN** labels canonicalize to `i64` indices deterministically and reuse the classifier label contract above.
