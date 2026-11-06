# Spec Delta

## Purpose

Defines `SKDBSCAN`: the region query, core/border/noise semantics, deterministic cluster
labeling, the prediction extension, representation parity and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKDBSCAN` SHALL be constructible only through its builder, exposing `eps` (default 0.5),
`minimum_samples` (default 5), `metric` (`Euclidean` default, `Manhattan`, `Chebyshev`),
`algorithm` (`Auto` default, `KdTree`, `BallTree`, `BruteForce`), `leaf_size` (default 40),
`backend` (default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses `eps = 0.5`, five samples, Euclidean distance, automatic search, leaf
  size 40, `Faer` and `Automatic`

### Requirement: Region query

Fit SHALL obtain each point's neighbourhood through a radius query: a tree for dense
metric-compatible data and brute force for sparse or mixed data. The neighbourhood SHALL
include the point itself.

#### Scenario: Tree and brute force agree

- **WHEN** the same dense data is fitted with `KdTree` and `BruteForce`
- **THEN** labels, core indices and cluster count are identical

#### Scenario: Sparse input uses brute force

- **WHEN** a sparse view is fitted with `Auto`
- **THEN** the brute-force searcher is used and the input is not densified

### Requirement: Core, border and noise semantics

A point SHALL be core when its neighbourhood contains at least `minimum_samples` points.
A non-core point within `eps` of a core point SHALL be a border point of that core point's
cluster; every other point SHALL carry the explicit `SKClusterLabel::Noise` variant.

#### Scenario: Core point seeds a cluster

- **WHEN** a point has at least `minimum_samples` neighbours within `eps`
- **THEN** it is a core point and starts or joins a cluster

#### Scenario: Border point joins a cluster

- **WHEN** a non-core point lies within `eps` of a core point
- **THEN** it receives that cluster's variant and is not noise

#### Scenario: Isolated point is noise

- **WHEN** a point has fewer than `minimum_samples` neighbours and is not within `eps` of
  any core point
- **THEN** its assignment is `SKClusterLabel::Noise`

### Requirement: Deterministic labeling

Cluster identifiers SHALL be assigned in the order of the smallest training index that
seeds each component, so repeated fits and thread counts produce identical labels.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** every label and cluster id is identical

#### Scenario: Thread count does not change labels

- **WHEN** fit runs with different rayon thread counts
- **THEN** labels and cluster count are identical

### Requirement: Model state

The model SHALL expose the typed assignments, the core-sample indices, the cluster count
and the fitted `eps`, and SHALL be safely shareable.

#### Scenario: Accessors describe the fit

- **WHEN** a fit completes
- **THEN** the assignment vector has one typed entry per training row, core indices are a
  subset of rows and the cluster count is the number of distinct cluster variants

### Requirement: Assignment extension

The fitted model SHALL implement `SKClustererPredictor<F>` by assigning a query to the
cluster of its nearest core sample when that sample lies within `eps`, otherwise to
`SKClusterLabel::Noise`.

#### Scenario: Query near a core point joins its cluster

- **WHEN** a query's nearest core sample is within `eps`
- **THEN** the query receives that sample's cluster variant

#### Scenario: Distant query is noise

- **WHEN** no core sample lies within `eps` of a query
- **THEN** the query receives the `Noise` variant

### Requirement: Representation parity

Fit and assignment SHALL accept dense and sparse views, dispatch once and never densify a
sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical data is fitted densely and as CSR
- **THEN** labels and cluster count agree

### Requirement: Validation

Fit SHALL reject empty input, non-positive `eps`, zero `minimum_samples`, feature-count
mismatch, non-finite values and an unsupported metric/algorithm pair.

#### Scenario: Non-positive eps is rejected

- **WHEN** `eps` is zero or negative
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Unsupported metric/algorithm pair is rejected

- **WHEN** `KdTree` is requested with Cosine distance
- **THEN** construction returns `InvalidHyperparameter` or
  `UnsupportedRepresentation` with guidance

### Requirement: Concurrent assignment

Fitted models SHALL produce the same assignment under concurrent calls as under a single
thread.

#### Scenario: One model serves several threads

- **WHEN** several threads assign with the same fitted model
- **THEN** every result equals the sequential result
