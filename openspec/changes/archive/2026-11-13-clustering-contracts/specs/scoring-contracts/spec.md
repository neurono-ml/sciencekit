# Spec Delta

## MODIFIED Requirements

### Requirement: Unsupervised scorer by assignments or by model
The unsupervised scorer SHALL expose a pure input over features and already-existing typed cluster assignments and SHALL provide an analogous convenient input that obtains the model's assignments before delegating.

#### Scenario: Silhouette-like over existing assignments
- **WHEN** pre-computed typed cluster assignments are provided with their features to the pure form
- **THEN** the score is computed without touching the model

#### Scenario: Convenient form uses cluster assignments
- **WHEN** a fitted clusterer is evaluated through the convenient form
- **THEN** the scorer obtains the model's typed assignments directly and never rounds a continuous response into an index

#### Scenario: Noise is handled explicitly
- **WHEN** the assignments contain the explicit noise variant
- **THEN** the scorer either excludes those rows or rejects them according to its documented policy, never treating noise as a cluster

#### Scenario: Assignment shape is validated
- **WHEN** the assignment vector length differs from the feature row count
- **THEN** the scorer returns a structured error from the central taxonomy
