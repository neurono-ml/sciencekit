# Spec Delta

## Purpose

Defines exact spatial indexes for nearest-neighbor search: how a tree is built from
training features, which metrics it supports, how exact `k`-nearest and radius queries
behave, and the validation, determinism and concurrency guarantees downstream estimators
and clustering rely on.

## ADDED Requirements

### Requirement: Immutable spatial index construction

The library SHALL build an immutable spatial index from a borrowed training feature
matrix, and the resulting index SHALL be safely shareable across threads without external
synchronization.

#### Scenario: Index is built once and queried many times

- **WHEN** one index is shared by several threads running queries simultaneously
- **THEN** every query completes without locking and returns the same result as a
  single-threaded query

#### Scenario: Building does not mutate the caller's data

- **WHEN** an index is built from a borrowed training view
- **THEN** the caller's buffer is unchanged after construction

### Requirement: Exact k-nearest-neighbor query

The library SHALL return exactly `k` nearest training rows for each query row, with
distances sorted in non-decreasing order and the corresponding training row indices.

#### Scenario: Query returns sorted neighbours

- **WHEN** a query row is compared against a fitted index with `k = 3`
- **THEN** the result contains three rows whose distances are sorted ascending

#### Scenario: A training row queried against its own index is its first neighbour

- **WHEN** a training row is used as a query against the index built from the same data
- **THEN** its first neighbour is itself with distance zero

#### Scenario: Every query row produces a result row

- **WHEN** a batch of `m` query rows is searched with `k`
- **THEN** the returned distances and indices have shape `(m, k)`

### Requirement: Deterministic tie ordering

The library SHALL break distance ties by the smallest training row index, so repeated
queries and parallel execution produce identical output.

#### Scenario: Equidistant candidates use index order

- **WHEN** two training rows are at the same distance from a query and only one fits in
  the result
- **THEN** the row with the smaller training index is returned

#### Scenario: Parallel equals sequential

- **WHEN** the same batch is queried sequentially and through the row-parallel path
- **THEN** both result matrices are identical

### Requirement: Radius query

The library SHALL return every training row within a caller-supplied radius for each
query row, together with its distance, without an upper bound on the number of neighbours.

#### Scenario: Radius includes the boundary

- **WHEN** a training row is exactly at the requested radius
- **THEN** it is included in the result

#### Scenario: Empty neighbourhoods are represented

- **WHEN** no training row lies within the radius
- **THEN** the query returns an empty neighbour list for that row, not an error

### Requirement: Dense-only spatial index

The spatial index SHALL accept dense training and query views only, and SHALL reject a
sparse view with `UnsupportedRepresentation` and guidance to the brute-force searcher.

#### Scenario: Sparse training input is rejected

- **WHEN** an index is built from a sparse CSR view
- **THEN** construction returns `UnsupportedRepresentation` before building any node

#### Scenario: Sparse query is rejected

- **WHEN** a sparse query view is passed to a fitted index
- **THEN** the query returns `UnsupportedRepresentation` and computes no distance

### Requirement: Metric catalogue and pruning compatibility

The spatial index SHALL support Euclidean, squared Euclidean, Manhattan and Chebyshev
distances, and SHALL reject a metric that is incompatible with tree pruning.

#### Scenario: Supported metric selects the matching geometry

- **WHEN** an index is built with Manhattan distance
- **THEN** every returned distance equals the Manhattan distance between the rows

#### Scenario: Non-metric distance is rejected

- **WHEN** an index is configured with cosine or NaN-aware Euclidean distance
- **THEN** construction returns `InvalidHyperparameter` naming the unsupported metric

### Requirement: Build validation

The library SHALL reject an empty training set, a zero leaf size, a zero neighbour count
and a neighbour count larger than the training-row count with structured errors.

#### Scenario: Empty training set is rejected

- **WHEN** an index is built from zero rows
- **THEN** construction returns `EmptyInput` naming the operation

#### Scenario: k larger than the training set is rejected

- **WHEN** a query asks for more neighbours than the index holds
- **THEN** the query returns a structured error instead of padding with sentinel rows

### Requirement: Query shape and finiteness validation

The library SHALL reject a query whose feature count differs from the training feature
count, and SHALL reject non-finite training or query values, before computing distances.

#### Scenario: Feature-count mismatch is rejected

- **WHEN** a query has a different feature count from the training data
- **THEN** the query returns `ShapeMismatch` with both dimensions

#### Scenario: Non-finite value is rejected

- **WHEN** training or query data contains NaN or infinity
- **THEN** the operation returns `NonFiniteValue` naming the position

### Requirement: Parallel batch queries

The library SHALL evaluate query rows in parallel with rayon while keeping per-row output
independent, and SHALL preserve exact results regardless of the thread count.

#### Scenario: Large batch uses the parallel path

- **WHEN** a batch large enough to cross the parallel grain is queried
- **THEN** the result equals the sequential result for the same input

#### Scenario: Concurrent batches do not interfere

- **WHEN** several threads query the same index with different batches
- **THEN** each thread receives the result it would receive alone
