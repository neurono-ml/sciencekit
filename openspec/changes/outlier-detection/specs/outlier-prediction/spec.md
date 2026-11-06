# Spec Delta

## Purpose

Defines the model-side outlier contract: the label convention, the decision function and
the anomaly score used by `SKIsolationForest` and later detectors.

## ADDED Requirements

### Requirement: Outlier labels

An outlier detector SHALL return `-1` for outliers and `1` for inliers, independent of the
feature scalar.

#### Scenario: Labels are canonical integers

- **WHEN** a fitted detector predicts over `f32` and `f64` features of identical values
- **THEN** both return identical `Array1<i64>` labels

#### Scenario: Label count follows the input

- **WHEN** a batch of `m` rows is predicted
- **THEN** the result has length `m`

### Requirement: Decision function

The detector SHALL expose a decision value where positive values indicate inliers and
negative values indicate outliers, and SHALL expose the threshold offset used to separate
them.

#### Scenario: Decision sign matches the label

- **WHEN** a row's decision value is negative
- **THEN** its label is `-1`; a non-negative value yields `1`

#### Scenario: Offset is exposed

- **WHEN** a fit completes
- **THEN** the model exposes the offset applied to the score

### Requirement: Anomaly score

The detector SHALL expose a score sample where larger values mean more normal and the
decision function is the score minus the offset.

#### Scenario: Score and decision agree

- **WHEN** a row's score and decision are requested
- **THEN** the decision equals the score minus the offset within the documented tolerance

### Requirement: Shareable models

Fitted detectors SHALL be safely shareable across threads and SHALL produce the same
labels, decisions and scores under concurrent calls.

#### Scenario: One model serves several threads

- **WHEN** several threads predict with the same fitted detector
- **THEN** every call completes without locking and returns its sequential result
