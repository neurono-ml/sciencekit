# scoring-contracts Specification

## Purpose
Defines the evaluation contracts: supervised and unsupervised scorers with dual input — pure form over already-existing predictions/assignments (no re-inference) and convenient provided form that runs inference and delegates — both fallible, serving as foundation for GridSearchCV and pipelines.

## Requirements

### Requirement: Supervised scorer with pure form and convenient form
The supervised scorer SHALL expose a pure input comparing true targets with already-existing predictions without running inference, and SHALL provide a second input receiving model, features and true targets, running inference and delegating to the pure form — without requiring additional implementation from the scorer author.

#### Scenario: Metric over stored predictions does not re-infer
- **WHEN** the consumer already holds a model's predictions and evaluates via the pure form
- **THEN** no inference runs and the metric value is produced directly from the comparison

#### Scenario: Convenient form runs inference and delegates
- **WHEN** a consumer evaluates a model via the convenient form
- **THEN** predictions are obtained from the model and the result equals the pure form applied to those same predictions

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

### Requirement: Evaluation is fallible by construction
Both scoring contracts SHALL return a fallible result, because the convenient form may fail at inference and pure forms may reject incoherent inputs.

#### Scenario: Predictions incoherent with targets produce a structured error
- **WHEN** the pure form receives predictions whose structure is not comparable with the true targets
- **THEN** the operation returns an error from the central taxonomy — never panics nor returns a numeric sentinel

### Requirement: Scorers are independent of the models they evaluate
Scoring contracts SHALL be generic over the evaluated model, allowing scorers reusable across algorithm families compatible with the same output shape.

#### Scenario: Same scorer evaluates models from distinct families
- **WHEN** two supervised models of different natures produce predictions comparable against the same targets
- **THEN** the same scorer evaluates them without additional adaptation

### Requirement: Probability scorer contract

The scoring contracts SHALL expose a probability scorer generic over a classifier model,
with a pure form over stored class probabilities and a convenient form that calls
`predict_probabilities` and delegates.

#### Scenario: Pure form over stored probabilities

- **WHEN** a probability scorer receives true targets and a probability matrix
- **THEN** it computes the metric without running inference

#### Scenario: Convenient form delegates

- **WHEN** a fitted classifier is scored through the convenient form
- **THEN** the scorer obtains `predict_probabilities` and delegates to the pure form

#### Scenario: Probability shape is validated

- **WHEN** the probability matrix has the wrong number of rows or columns
- **THEN** the scorer returns `ShapeMismatch`
