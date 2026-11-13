# Spec Delta

## Purpose

Defines `SKLocalOutlierFactor` (reachability distance, local reachability density, local
outlier factor, fit-only versus novelty workflows) and `SKOneClassSVM` (one-class SMO
fitting over the shared kernel catalogue, `nu`, decision function and typed labels) in
`sciencekit_outlier`, including their representation parity, export/import contract and
validation.

## ADDED Requirements

### Requirement: Builder-only LOF construction with documented defaults

`SKLocalOutlierFactor` SHALL be constructible only through its builder, exposing
`nearest_neighbors_count` (default 20), `contamination` (`Auto` default or a fraction in
`(0, 0.5]`), `novelty` (default false), `metric` (default Euclidean), `backend` (default
`Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an LOF detector is built without calling any setter
- **THEN** it uses 20 neighbours, automatic contamination, fit-only mode, Euclidean
  distance, `Faer` and `Automatic`

#### Scenario: Squared Euclidean is rejected

- **WHEN** the squared Euclidean distance is configured
- **THEN** construction returns `InvalidHyperparameter` naming the accepted metrics

#### Scenario: Backend is recorded

- **WHEN** a non-default backend kind is configured and the model fits and predicts
- **THEN** every operation span records that configured kind

### Requirement: Reachability distance and local reachability density

The detector SHALL compute the core distance to the k-th nearest neighbour, the
reachability distance as the maximum of the neighbour's core distance and the direct
distance, and the local reachability density as the inverse of the mean reachability
distance to the k neighbours.

#### Scenario: Reachability is at least the direct distance

- **WHEN** any pair of points is evaluated
- **THEN** its reachability distance is greater than or equal to its direct distance

#### Scenario: Dense cluster has higher density

- **WHEN** a point inside a tight cluster and a point in a sparse region are evaluated with
  the same neighbour count
- **THEN** the cluster point has the higher local reachability density

#### Scenario: Duplicate points do not produce infinities

- **WHEN** more than `k` points coincide
- **THEN** the affected density and factor are finite, using the documented zero-density
  rule

### Requirement: Local outlier factor and negative outlier factor

The detector SHALL compute the local outlier factor as the mean neighbour density divided
by the point density and SHALL expose its negation as the normality score, so larger values
mean more normal.

#### Scenario: Isolated point scores higher than cluster point

- **WHEN** an isolated point and a point inside a dense cluster are scored
- **THEN** the isolated point's local outlier factor is higher and its negative outlier
  factor is lower

#### Scenario: Uniform density has unit factor

- **WHEN** every point has the same neighbourhood density
- **THEN** every local outlier factor is `1` within tolerance

### Requirement: Fit-only and novelty workflows

The detector SHALL support a fit-only workflow that scores only the training data and a
novelty workflow that scores new data, refusing out-of-sample prediction in the fit-only
workflow with a typed error.

#### Scenario: Fit-only exposes training scores

- **WHEN** a detector with `novelty = false` is fitted
- **THEN** `fit_predict` returns one label per training row and the negative outlier factor
  vector is available

#### Scenario: Fit-only refuses new data

- **WHEN** `predict`, `decision_function` or `score_samples` is called on a fit-only model
- **THEN** it returns `UnsupportedOperation` suggesting `novelty(true)`, without panicking

#### Scenario: Novelty scores new data

- **WHEN** a detector with `novelty = true` is fitted and a new row is scored
- **THEN** the row's neighbours, density and factor are computed against the training data
  and a typed label is returned

### Requirement: LOF contamination offset and labels

The detector SHALL derive the decision offset from `contamination`, expose it, and return
`SKOutlierLabel::Outlier` for negative decision values and `SKOutlierLabel::Inlier`
otherwise.

#### Scenario: Explicit contamination selects a quantile

- **WHEN** `contamination` is a fraction
- **THEN** the offset is the corresponding quantile of the fit-time normality scores

#### Scenario: Decision sign matches the label

- **WHEN** a row's decision value is negative
- **THEN** its label is `Outlier`; a non-negative value yields `Inlier`

### Requirement: LOF representation parity

LOF fit and prediction SHALL accept dense, sparse and mixed inputs, dispatch on the
representation exactly once per operation, use the dense spatial index for dense training
data and the brute-force searcher for sparse training or query data, and never densify a
sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical data is fitted densely and as CSR with the same configuration
- **THEN** factors, decisions and labels agree within the documented tolerance

#### Scenario: Sparse training never densifies

- **WHEN** a CSR matrix is fitted
- **THEN** the brute-force branch reads only stored entries and no dense copy is allocated

#### Scenario: Mixed novelty query is supported

- **WHEN** a sparse row queries a dense training model, or a dense row queries a sparse
  training model
- **THEN** the request succeeds through the brute-force searcher

### Requirement: Builder-only One-Class SVM construction

`SKOneClassSVM` SHALL be constructible only through its builder, exposing `nu` (default
0.5) in `(0, 1]`, `kernel` (default radial basis function), `gamma` (default `Scale`),
`degree` (default 3), `coef0` (default 0), `tolerance`, `maximum_number_of_iterations`,
`cache_size`, `shrinking` (default true), `backend` (default `Faer`) and `execution_mode`
(default `Automatic`).

#### Scenario: Default configuration

- **WHEN** a One-Class SVM is built without calling any setter
- **THEN** it uses `nu = 0.5`, an RBF kernel with scale gamma, `Faer` and `Automatic`

#### Scenario: Nu outside the open-closed interval is rejected

- **WHEN** `nu` is zero, negative or greater than one
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Precomputed kernel is rejected

- **WHEN** the precomputed kernel is configured
- **THEN** construction returns `InvalidHyperparameter` explaining that the estimator has no
  precomputed training kernel

### Requirement: One-Class SVM kernel selection

The estimator SHALL support the linear, polynomial, radial-basis-function and sigmoid
kernels from the shared catalogue, with the catalogue's gamma policy and validated degree
and offset parameters.

#### Scenario: Every supported kernel fits

- **WHEN** each of the four kernels is configured on a small fixture
- **THEN** fit converges and every row receives a typed label

#### Scenario: Invalid kernel parameters are rejected

- **WHEN** the polynomial degree is zero or `gamma` is negative
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Kernel parity with the catalogue

- **WHEN** the same kernel parameters are used through `SKKernelParameters` and through the
  estimator
- **THEN** the evaluated kernel values agree within the documented tolerance

### Requirement: One-Class SVM fitting and decision function

The estimator SHALL solve the one-class dual with the shared SMO primitives, derive
support vectors, dual coefficients and `rho`, and expose a decision function where negative
values indicate outliers.

#### Scenario: Bounded dual solution

- **WHEN** fit converges
- **THEN** every dual coefficient lies in `[0, 1/(nu · n)]` and the coefficients sum to one

#### Scenario: Decision depends on support vectors only

- **WHEN** the decision value of a row is computed
- **THEN** it equals the support-vector kernel expansion minus `rho` within tolerance

#### Scenario: Non-separable data still fits

- **WHEN** the training data contains overlapping clusters
- **THEN** fit converges within the iteration budget and returns a finite `rho`

#### Scenario: Exhausted budget is reported

- **WHEN** the iteration budget is too small to converge
- **THEN** fit returns `NotConverged`

### Requirement: One-Class SVM representation parity

One-Class SVM fit and prediction SHALL accept dense, sparse and mixed inputs through the
kernel catalogue without densifying a sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical data is fitted densely and as CSR with the same configuration
- **THEN** support-vector decisions and labels agree within the documented tolerance

#### Scenario: Sparse input is not densified

- **WHEN** a CSR matrix is fitted or predicted
- **THEN** only stored entries are read by the kernel functions

### Requirement: Outlier predictor contract

Both detectors SHALL implement `SKOutlierPredictor<F>`, returning `SKOutlierLabel` values,
shifted decisions and normality scores with the documented offset.

#### Scenario: Score and decision agree

- **WHEN** a row's score and decision are requested
- **THEN** the decision equals the score minus the offset within tolerance

#### Scenario: Labels are typed

- **WHEN** either detector predicts
- **THEN** the result is an array of `SKOutlierLabel`, never raw `-1`/`1` integers

#### Scenario: Batch shapes

- **WHEN** `m` rows are predicted
- **THEN** labels, decisions and scores all have length `m`

### Requirement: Export and import

Both fitted models SHALL implement `SKExportable` and `SKImportable`, round-tripping every
configuration and fitted tensor, preserving the training representation and returning the
same predictions after import.

#### Scenario: Dense round-trip preserves predictions

- **WHEN** a densely fitted detector is exported and imported
- **THEN** scores, decisions and labels match the original within tolerance

#### Scenario: Sparse round-trip does not densify

- **WHEN** a sparsely fitted detector is exported and imported
- **THEN** the state records the CSR representation and the imported model agrees without a
  dense training copy

#### Scenario: Metadata mismatch is rejected

- **WHEN** a state exported by one estimator is imported by the other
- **THEN** import returns a structured metadata error

### Requirement: Validation, determinism and concurrency

Both detectors SHALL reject empty input, invalid neighbour counts, feature-count mismatch,
non-finite values and prediction before fit, SHALL be deterministic for a fixed
configuration and SHALL be safely shareable across threads.

#### Scenario: Empty input is rejected

- **WHEN** fit receives zero rows
- **THEN** it returns `EmptyInput`

#### Scenario: Too few rows for the neighbour count is rejected

- **WHEN** `nearest_neighbors_count` is not smaller than the number of distinct training
  rows
- **THEN** LOF fit returns `InvalidHyperparameter`

#### Scenario: Feature-count mismatch is rejected

- **WHEN** prediction receives a row with a different feature count from fit
- **THEN** it returns `ShapeMismatch`

#### Scenario: Repeated fits agree

- **WHEN** the same detector fits the same data twice
- **THEN** scores, decisions, labels and offsets are identical

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted model
- **THEN** every call returns its sequential result
