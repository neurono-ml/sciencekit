# mini-batch-k-means Specification

## Purpose
Defines `SKMiniBatchKMeans`: batch center updates with per-center learning rates,
initialization, `partial_fit`, streaming fitting, epoch handling, reassignment, model
state, assignment prediction and validation.

## Requirements

### Requirement: Builder-only construction with documented defaults

`SKMiniBatchKMeans` SHALL be constructible only through its builder, exposing the
documented cluster, initialization, batch, iteration, epoch, reassignment, backend and
execution-mode setters.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses eight clusters, k-means++, three initializations, batches of 1024,
  100 iterations, one streaming epoch, no reassignment-ratio change, `Faer` and `Automatic`

### Requirement: Batch center update

`partial_fit` SHALL assign every batch row to its nearest center and update each center by
the per-center learning rate `1 / count[center]`, incrementing the center count.

#### Scenario: One-center closed form

- **WHEN** a batch is assigned entirely to one center
- **THEN** that center moves to the mean of the batch rows and its count increases by the
  batch size

#### Scenario: Learning rate decays with the count

- **WHEN** a center has received `n` rows and a new row is assigned to it
- **THEN** the update weight is `1 / (n + 1)`

### Requirement: Initialization

`fit` and `fit_streaming` SHALL initialize centers from at most
`initialization_batch_size` rows using the configured initializer, and SHALL run
`number_of_initializations` independent seeded initializations, retaining the best inertia.

#### Scenario: Initialization sample respects the bound

- **WHEN** `initialization_batch_size` is smaller than the dataset
- **THEN** initialization uses at most that many rows

#### Scenario: Best initialization wins

- **WHEN** multiple initializations are configured
- **THEN** the model with the smaller inertia is retained

### Requirement: Partial fit across batches

Calling `partial_fit` repeatedly SHALL preserve the centers, counts and convergence state
between calls, and the result SHALL equal one `fit` over the concatenated batches within
the documented tolerance.

#### Scenario: Batched equals full fit

- **WHEN** the same data is fitted in one call and through repeated `partial_fit` calls
- **THEN** centers and assignments agree within the documented tolerance

#### Scenario: State persists

- **WHEN** `partial_fit` is called twice
- **THEN** the second call starts from the first call's centers and counts

### Requirement: Streaming fit

`fit_streaming` SHALL drive an `SKSourceProvider` through the shared prefetch executor,
supporting multiple epochs when the provider is reopenable and stopping after
`maximum_number_of_epochs` or convergence.

#### Scenario: One-pass streaming

- **WHEN** a non-reopenable provider is used
- **THEN** exactly one pass is consumed and the model is returned

#### Scenario: Multi-epoch streaming

- **WHEN** a reopenable provider is used with more than one epoch
- **THEN** the source is reopened per epoch and the model records the epochs consumed

### Requirement: Reassignment

When a center's count is below `reassignment_ratio × mean_count`, the estimator SHALL
reassign the point with the largest distance to its current center to that center,
deterministically, and reset its count.

#### Scenario: Low-count center is reassigned

- **WHEN** a center falls below the reassignment threshold
- **THEN** a distant point is moved to it and the fit continues

#### Scenario: Reassignment is deterministic

- **WHEN** the same batches and seed trigger the reassignment
- **THEN** the same point is chosen

### Requirement: Model state and assignment

The model SHALL expose centers, per-center counts, typed training assignments when fitted
in memory, inertia, batches processed and epochs consumed, and SHALL implement
`SKClustererPredictor<F>` plus `transform`. It never emits `Noise`.

#### Scenario: State accessors

- **WHEN** a fit completes
- **THEN** the center and count vectors have length `number_of_clusters`

#### Scenario: Assignment uses the nearest center

- **WHEN** a dense or sparse batch is assigned
- **THEN** each row receives the typed cluster variant of its nearest center

### Requirement: Representation parity

`partial_fit`, `fit`, `fit_streaming` and assignment SHALL accept dense and sparse data,
dispatch once and never densify a sparse batch.

#### Scenario: Sparse batch equals dense

- **WHEN** the same logical batch is supplied densely and as CSR
- **THEN** the updated centers agree within the documented tolerance

### Requirement: Validation

The estimator SHALL reject an empty batch, zero clusters, a batch smaller than the cluster
count, invalid initialization size, invalid reassignment ratio, non-finite values and
feature-count mismatch.

#### Scenario: Small batch is rejected

- **WHEN** a batch has fewer rows than `number_of_clusters`
- **THEN** `partial_fit` returns `InvalidHyperparameter`

#### Scenario: Reassignment ratio out of range is rejected

- **WHEN** `reassignment_ratio` is outside `[0, 1]`
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Determinism and concurrency

A fixed seed and input SHALL produce identical models across repeated runs and thread
counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated runs agree

- **WHEN** the same estimator fits the same data twice
- **THEN** centers, counts and assignments are identical

#### Scenario: Concurrent assignment agrees

- **WHEN** several threads assign with the same fitted model
- **THEN** every result equals the sequential result
