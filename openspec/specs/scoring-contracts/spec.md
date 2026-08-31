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
The unsupervised scorer SHALL expose a pure input over features and already-existing assignments/outputs (e.g.: cluster labels) and SHALL provide an analogous convenient input that obtains the model's outputs before delegating.

#### Scenario: Silhouette-like over existing assignments
- **WHEN** pre-computed cluster assignments are provided with their features to the pure form
- **THEN** the score is computed without touching the model

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

