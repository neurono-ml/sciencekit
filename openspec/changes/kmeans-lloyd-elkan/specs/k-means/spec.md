# Spec Delta

## Purpose

Defines `SKKMeans`: initialization, Lloyd and Elkan assignment, best-of-initializations
selection, empty-cluster handling, model state, assignment prediction, representation
parity, determinism and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKKMeans` SHALL be constructible only through its builder, exposing `number_of_clusters`
(default 8), `initialization` (`KMeansPlusPlus` default), `number_of_initializations`
(default 10 for random, 1 for k-means++), `maximum_number_of_iterations` (default 300),
`tolerance` (default 1e-4), `algorithm` (`Auto` default, `Lloyd`, `Elkan`),
`random_seed` (default 0), `backend` (default `Faer`) and `execution_mode` (default
`Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses eight clusters, k-means++, automatic algorithm selection, 300 iterations,
  tolerance `1e-4`, `Faer` and `Automatic`

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the estimator is fitted
- **THEN** the fitted model records that backend in its assignment spans

### Requirement: K-means++ initialization

The k-means++ initializer SHALL choose the first center uniformly at random and each next
center with probability proportional to the squared distance to the nearest chosen center,
using the seeded generator.

#### Scenario: Centers are distinct

- **WHEN** k-means++ initializes `k` centers on data with at least `k` distinct points
- **THEN** all chosen centers are distinct training points

#### Scenario: Initialization is reproducible

- **WHEN** the same seed and data are initialized twice
- **THEN** the chosen centers are identical

### Requirement: Lloyd assignment and update

Lloyd fitting SHALL alternate assigning every row to its nearest center and moving each
center to the mean of its assigned rows until the maximum center movement falls to or below
`tolerance` or the iteration budget is exhausted.

#### Scenario: Two well-separated clusters converge

- **WHEN** two separated blobs are fitted with two clusters
- **THEN** the labels separate the blobs and inertia is near zero

#### Scenario: Exhausted budget returns a model

- **WHEN** the iteration budget is exhausted before the tolerance is reached
- **THEN** the model is returned with `converged() == false` and its iteration count, not a
  non-convergence error

### Requirement: Elkan acceleration is exact

The Elkan algorithm SHALL produce the same assignments and inertia as Lloyd within the
documented tolerance, using triangle-inequality bounds and never evaluating an impossible
distance.

#### Scenario: Elkan equals Lloyd

- **WHEN** the same data and initial centers are fitted with `Elkan` and `Lloyd`
- **THEN** labels, centers and inertia agree within the documented tolerance

#### Scenario: Elkan selects only Euclidean-compatible data

- **WHEN** Elkan is requested on a representation it cannot bound
- **THEN** the estimator resolves to Lloyd or returns a structured error, never silently
  changing the result

### Requirement: Best-of-initializations selection

When `number_of_initializations > 1`, fit SHALL run independent seeded initializations and
retain the model with the smallest inertia, breaking exact ties by the earliest
initialization index.

#### Scenario: Lowest inertia wins

- **WHEN** two initializations produce different inertia values
- **THEN** the model with the smaller inertia is returned

#### Scenario: Tie is deterministic

- **WHEN** two initializations produce the same inertia
- **THEN** the earlier initialization index is returned

### Requirement: Empty-cluster repair

If an update leaves a cluster with no assigned rows, fit SHALL reassign a deterministic
farthest point to that center before continuing.

#### Scenario: Empty cluster is repaired

- **WHEN** a cluster receives no rows in an update
- **THEN** a training point is reassigned to it and the fit continues

#### Scenario: Repair is deterministic

- **WHEN** the same data and seed trigger the repair
- **THEN** the same point is reassigned

### Requirement: Fitted model state and accessors

The model SHALL expose the cluster centers, the training labels, the inertia, the number of
iterations executed and the convergence flag.

#### Scenario: Accessors describe the fit

- **WHEN** a fit completes
- **THEN** `cluster_centers` has one row per cluster, `labels` has one entry per training
  row and `inertia` equals the sum of squared distances to the assigned centers

### Requirement: Cluster assignment prediction

The fitted model SHALL implement `SKClustererPredictor<F>`, returning canonical `i64`
assignments in `0..number_of_clusters` for dense and sparse inputs.

#### Scenario: New rows are assigned to the nearest center

- **WHEN** a dense or sparse batch is assigned
- **THEN** each row receives the index of its nearest center

#### Scenario: Assignment is scalar-independent

- **WHEN** the same logical batch is `f32` and `f64`
- **THEN** the assignments are identical

### Requirement: Distance-to-centers transform

The model SHALL expose `transform`, returning the Euclidean distance from every row to
every center.

#### Scenario: Transform shape and values

- **WHEN** `m` rows are transformed on a `k`-cluster model
- **THEN** the result is `m × k` and a row's smallest value is its assigned center's
  distance

### Requirement: Representation parity

Fit and assignment SHALL accept dense and sparse views, dispatch once and never densify a
sparse input.

#### Scenario: Sparse equals dense

- **WHEN** the same logical design is fitted densely and as CSR with the same seed
- **THEN** labels, centers and inertia agree within the documented tolerance

#### Scenario: Sparse input is not densified

- **WHEN** a wide sparse design is fitted
- **THEN** no dense copy of the design is created

### Requirement: Validation

Fit SHALL reject empty input, zero clusters, more clusters than samples, non-finite values,
invalid tolerance/iteration/initialization parameters and feature-count mismatch.

#### Scenario: More clusters than samples is rejected

- **WHEN** `number_of_clusters` exceeds the training-row count
- **THEN** fit returns `InvalidHyperparameter`

#### Scenario: Non-finite value is rejected

- **WHEN** fit or assignment receives NaN or infinity
- **THEN** the operation returns `NonFiniteValue` naming the position

### Requirement: Determinism and concurrency

A fixed seed and input SHALL produce identical models and assignments across repeated fits
and thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** centers, labels and inertia are identical

#### Scenario: Concurrent assignment agrees

- **WHEN** several threads assign with the same fitted model
- **THEN** every result equals the sequential result
