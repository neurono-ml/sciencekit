# Spec Delta

## Purpose

Defines the model-side outlier contract: the typed label convention, the decision function
and the anomaly score used by `SKIsolationForest` and later detectors.

## ADDED Requirements

### Requirement: Typed outlier labels

An outlier detector SHALL return one typed label per input row, either the explicit
`Inlier` or `Outlier` variant, independent of the feature scalar.

#### Scenario: Labels are typed variants

- **WHEN** a fitted detector predicts over `f32` and `f64` features of identical values
- **THEN** both return identical typed label arrays

#### Scenario: Label count follows the input

- **WHEN** a batch of `m` rows is predicted
- **THEN** the result has length `m`

#### Scenario: Label size is bounded

- **WHEN** the label type is inspected
- **THEN** it occupies at most one byte per element

### Requirement: Interoperability conversion

The library SHALL provide explicit conversions between the typed label and the `i64`
convention used by external tools, mapping `Outlier` to `-1` and `Inlier` to `1`.

#### Scenario: Outlier maps to -1

- **WHEN** an `Outlier` label is converted for external interop
- **THEN** it becomes `-1`

#### Scenario: Inlier maps to 1

- **WHEN** an `Inlier` label is converted for external interop
- **THEN** it becomes `1`

#### Scenario: Invalid external value is rejected

- **WHEN** an external `i64` other than `-1` or `1` is converted into the typed label
- **THEN** the conversion returns a structured error

### Requirement: Decision function

The detector SHALL expose a decision value where positive values indicate inliers and
negative values indicate outliers, and SHALL expose the threshold offset used to separate
them.

#### Scenario: Decision sign matches the label

- **WHEN** a row's decision value is negative
- **THEN** its label is `Outlier`; a non-negative value yields `Inlier`

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
