# Spec Delta

## Purpose

Defines the k-nearest-neighbor classifier and regressor: how the estimator chooses a
search algorithm at runtime, how neighbours vote or average, how probabilities and ties
are resolved, which representations are accepted, and how validation and concurrency
behave.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing
`nearest_neighbors_count` (default 5), `weights` (`Uniform` default), `algorithm` (`Auto`
default), `metric` (`Euclidean` default), `leaf_size` (default 40), `backend` (default
`Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses five uniform neighbours, automatic search, Euclidean distance, leaf
  size 40, `Faer` and `Automatic`

#### Scenario: Backend and execution mode reach the model

- **WHEN** a non-default backend and execution mode are configured and the estimator is
  fitted
- **THEN** the fitted model records those values in its prediction spans

### Requirement: Runtime search algorithm selection

The estimator SHALL accept `Auto`, `KdTree`, `BallTree` and `BruteForce`, and `Auto` SHALL
choose a tree for dense Euclidean, Manhattan or Chebyshev data and brute force for sparse,
mixed or cosine data. An explicit algorithm that cannot serve the input representation
SHALL return `UnsupportedRepresentation` with guidance.

#### Scenario: Auto uses a tree on dense Euclidean data

- **WHEN** a dense Euclidean fit completes with `Auto`
- **THEN** the model reports a tree-based search strategy

#### Scenario: Auto uses brute force on sparse data

- **WHEN** a sparse fit completes with `Auto`
- **THEN** the model reports the brute-force strategy and succeeds without densifying

#### Scenario: Explicit incompatible algorithm is rejected

- **WHEN** `KdTree` is requested for sparse training data
- **THEN** fit returns `UnsupportedRepresentation` and computes nothing

### Requirement: Classification by neighbour vote

The classifier SHALL predict the class with the most votes among the `k` nearest
training rows, weighting each neighbour uniformly or by the inverse of its distance.

#### Scenario: Majority vote

- **WHEN** the five nearest neighbours of a row belong to classes `[0, 0, 0, 1, 1]` with
  uniform weights
- **THEN** the predicted label is `0`

#### Scenario: Distance weighting changes the vote

- **WHEN** the three nearest neighbours are class `1` and the next two are class `0`, and
  `Distance` weighting is enabled
- **THEN** the closer class-`1` neighbours dominate the vote

### Requirement: Class probability estimates

The classifier SHALL expose one probability per class, equal to the weighted neighbour
fraction for that class, with rows summing to one.

#### Scenario: Probabilities are neighbour fractions

- **WHEN** three of five uniform-weight neighbours belong to class `0`
- **THEN** the probability of class `0` is `0.6`

#### Scenario: All classes appear in the probability matrix

- **WHEN** a class is present in the fitted label table but not among a row's neighbours
- **THEN** its probability is zero and the row still sums to one

### Requirement: Regression by neighbour average

The regressor SHALL predict the mean of the neighbour responses, or the inverse-distance
weighted mean when `Distance` weighting is enabled.

#### Scenario: Uniform mean

- **WHEN** the three nearest responses are `2.0`, `4.0` and `6.0`
- **THEN** the uniform prediction is `4.0`

#### Scenario: Distance-weighted mean

- **WHEN** the nearest response is at distance zero and the next is farther away
- **THEN** the weighted prediction is dominated by the closer response and stays finite

### Requirement: Deterministic tie-breaking

The estimator SHALL break distance ties by the smallest training row index and class
ties by the smallest canonical class index, so results do not depend on thread count.

#### Scenario: Equidistant neighbours use index order

- **WHEN** more candidates than `k` are at the same distance
- **THEN** the candidates with the smallest training indices are selected

#### Scenario: Parallel prediction equals sequential prediction

- **WHEN** a large batch is predicted through the parallel path and sequentially
- **THEN** both results are identical

### Requirement: Representation parity for the estimator

`SKKNeighborsClassifier` and `SKKNeighborsRegressor` SHALL accept dense, sparse and mixed
dense/sparse fit and predict inputs, dispatch once per operation and never densify a
sparse input.

#### Scenario: Sparse fit and prediction

- **WHEN** the same logical data is fitted densely and as CSR
- **THEN** both models produce the same predictions within the documented tolerance

#### Scenario: Mixed prediction

- **WHEN** a model fitted on dense data predicts on a sparse batch, or the reverse
- **THEN** prediction succeeds through the brute-force path without densifying the sparse
  side

### Requirement: Target canonicalization

The classifier SHALL accept integer and nominal targets through the shared label table,
and SHALL order classes by the canonical table order rather than by observation order.

#### Scenario: Nominal labels are canonicalized

- **WHEN** nominal targets are fitted
- **THEN** predictions are returned as canonical label indices and the class table is
  available on the model

#### Scenario: Integer labels preserve their values

- **WHEN** integer targets are fitted
- **THEN** predicted labels match the original integer values through the table

### Requirement: Validation

The estimator SHALL reject empty input, a zero or excessive neighbour count, a
feature-count mismatch, non-finite values, sparse NaN input and an unsupported
metric/algorithm pair with structured central errors before computing neighbours.

#### Scenario: Excessive neighbour count is rejected

- **WHEN** `nearest_neighbors_count` exceeds the training-row count
- **THEN** fit or predict returns a structured error instead of padding results

#### Scenario: Feature-count mismatch is rejected

- **WHEN** a prediction batch has a different feature count from the fitted data
- **THEN** prediction returns `ShapeMismatch` with both dimensions

#### Scenario: Non-finite value is rejected

- **WHEN** fit or prediction receives NaN or infinity
- **THEN** the operation returns `NonFiniteValue` naming the position

### Requirement: Concurrent prediction

Fitted models SHALL be safely shareable across threads and SHALL produce the same result
under concurrent prediction as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads predict with the same fitted model
- **THEN** every call completes without locking and returns its sequential result
