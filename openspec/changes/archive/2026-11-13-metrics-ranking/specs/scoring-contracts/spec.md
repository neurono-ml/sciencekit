# Spec Delta

## ADDED Requirements

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
