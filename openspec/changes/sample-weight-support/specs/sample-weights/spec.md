# sample-weights Specification

## Purpose

Defines the per-sample weight contract shared by estimators, streaming batches and metrics,
so every algorithm that admits weights consumes them the same way and unweighted behavior
is preserved exactly.

## ADDED Requirements

### Requirement: Optional weights on fitting operations

Every estimator whose mathematics admits per-sample weights SHALL accept an optional
`sample_weight` on `fit`, `partial_fit` and `fit_streaming`; omitting it SHALL produce
exactly the unweighted result.

#### Scenario: Unweighted equals weighted with ones

- **WHEN** an estimator is fitted twice, once without weights and once with all weights set
  to one
- **THEN** the two models are identical within the documented tolerance

#### Scenario: Zero-weight rows do not contribute

- **WHEN** a row carries weight zero
- **THEN** it does not contribute to the fitted statistics (means, counts, inertias) as if
  the row were absent

### Requirement: Weight validation

Weights SHALL be finite and non-negative, with one weight per row; violations fail with the
central error taxonomy before any statistics are accumulated.

#### Scenario: Negative weight is rejected

- **WHEN** a weight vector contains a negative value
- **THEN** the fit fails with the invalid-value error naming the row range

#### Scenario: Length mismatch is a shape error

- **WHEN** the weight vector length differs from the row count
- **THEN** the fit fails with the shape-mismatch variant carrying both lengths

### Requirement: Streaming batches carry weights

`SKDataBatch` and `SKSupervisedBatch` SHALL carry an optional weight vector whose length
matches the batch row count, and the streaming double-buffer guard SHALL account for it.

#### Scenario: Weighted streaming fit

- **WHEN** a streaming fit receives weighted batches
- **THEN** the fitted model equals the in-memory weighted fit within the documented
  tolerance

#### Scenario: Weightless batches are unchanged

- **WHEN** a streaming pipeline passes batches without weights
- **THEN** memory accounting and behavior are exactly as before

### Requirement: Composition with class weights and metrics

Sample weights SHALL compose multiplicatively with class weights, and the metrics SHALL
expose weighted scoring entry points consuming the same weights.

#### Scenario: Sample and class weights compose

- **WHEN** both class weights and sample weights are configured
- **THEN** the effective weight of a row is the product, and the tree/SGD weighted
  statistics reflect it

#### Scenario: Weighted metric matches the weighted fit

- **WHEN** a weighted fit is scored with the same weights
- **THEN** the weighted accuracy equals the weights-normalized rate used by the estimator's
  own statistics
