# standard-scaler Specification

## Purpose
Defines `SKStandardScaler`, the first Wave 1 preprocessing estimator: a builder-configured,
backend-carrying standardizer with a distinct fitted model, reachable identically from a
dense `fit`, an incremental `partial_fit` and an out-of-core `fit_streaming`.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKStandardScaler` SHALL be constructed only through `SKStandardScaler::build()`, which
SHALL expose chainable `with_mean(bool)` and `with_std(bool)` (both defaulting to `true`),
`backend(SKBackendKind)` (defaulting to `SKBackendKind::Faer`) and
`execution_mode(SKExecutionMode)` (defaulting to `Automatic`); the direct constructor SHALL
stay private.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it centers and scales (`with_mean` and `with_std` true), carries
  `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Configuration override flows through the model

- **WHEN** an estimator is built with `with_mean(false)` and a non-default backend
- **THEN** the fitted model reports a zero mean, leaves values uncentered, and carries that
  backend kind

### Requirement: Dense fit learns per-column mean and population scale

`fit` SHALL learn, for each column, the arithmetic mean and the population standard
deviation (dividing the sum of squared deviations by the row count), returning a fitted
model that exposes them. The estimator SHALL stay generic over the scalar `F: SKFloat`
(both `f32` and `f64`).

#### Scenario: Mean and scale on a hand-checkable matrix

- **WHEN** `[[1, 2], [3, 4]]` is fitted with the default configuration
- **THEN** the model reports mean `[2, 3]` and scale `[1, 1]`

#### Scenario: Both supported floats

- **WHEN** the same data is fitted as `f32` and as `f64`
- **THEN** both fits succeed and report the same statistics in their own scalar

### Requirement: Transform produces z-scores honoring the masking flags

`transform` SHALL center each value by its column mean (unless `with_mean` is false) and
divide by its column scale (unless `with_std` is false), returning an owned array of the
same shape; it SHALL be generic over `F` and SHALL use the model's learned statistics only.

#### Scenario: Centering and scaling

- **WHEN** the model fitted on `[[1, 2], [3, 4]]` transforms the same data
- **THEN** the output is `[[-1, -1], [1, 1]]`

#### Scenario: Masking disables an axis

- **WHEN** `with_mean` is false, transform SHALL NOT center; when `with_std` is false, it
  SHALL NOT scale
- **THEN** the corresponding operation is the identity on that axis

### Requirement: Zero-variance columns stay finite

A column with non-positive variance (a constant column, or one whose computed variance
rounds to zero) SHALL receive a unit scale so that its transformed values are zero (when
centering) instead of `NaN` or infinity.

#### Scenario: Constant column

- **WHEN** a matrix containing a constant column is fitted and transformed
- **THEN** that column's scale is one and its transformed values are zero

### Requirement: Structured errors on invalid input

The estimator SHALL reject empty inputs and sparse inputs before processing any element,
and `transform` SHALL reject a column count differing from the fitted model, each with the
central `SKError` taxonomy (`ShapeMismatch`, `UnsupportedRepresentation`).

#### Scenario: Empty input

- **WHEN** `fit` receives a matrix with zero rows or zero columns
- **THEN** it returns a `ShapeMismatch` error before computing anything

#### Scenario: Sparse input

- **WHEN** `fit` receives a sparse matrix
- **THEN** it returns an `UnsupportedRepresentation` error at the dense-only boundary

#### Scenario: Transform column mismatch

- **WHEN** a fitted model transforms data whose column count differs from its own
- **THEN** it returns a `ShapeMismatch` error and produces no output

### Requirement: Incremental partial_fit converges to the dense result

The fitted model SHALL accept `partial_fit` over additional batches, merging them into
running moments with an order-independent, exact-up-to-rounding update, so that fitting a
prefix and then `partial_fit`-ing the rest equals the dense fit over the whole data;
`sample_count` SHALL accumulate across `fit` and every `partial_fit`. An empty batch SHALL
be a no-op; a column-count mismatch SHALL fail without touching the model.

#### Scenario: Prefix plus partial_fit equals dense

- **WHEN** a model is fitted on part of the data and `partial_fit` on the remainder
- **THEN** its mean, scale and sample count equal the dense fit's within tolerance

#### Scenario: Single-row accumulation

- **WHEN** a model is fitted on one row and `partial_fit`-ed row by row
- **THEN** the final statistics equal the dense fit's

#### Scenario: Empty batch is a no-op

- **WHEN** `partial_fit` receives a batch with zero rows
- **THEN** neither the statistics nor the sample count change

#### Scenario: Mismatched batch is rejected

- **WHEN** `partial_fit` receives a batch with the wrong column count
- **THEN** it returns a `ShapeMismatch` error and the model is unchanged

### Requirement: Out-of-core fit_streaming equals the dense fit

`fit_streaming` SHALL consume a sequential `SKLazySource` and produce the same model the
dense `fit` would produce for the concatenated data: under an in-memory execution intent
it SHALL collect the batches and reuse the dense path; otherwise it SHALL accumulate
per-batch moments through the streaming executor. An empty source or a batch sequence with
inconsistent column counts SHALL fail with a structured error.

#### Scenario: Uneven batches equal dense

- **WHEN** the data is streamed in uneven batches and compared with the dense fit
- **THEN** the mean, scale and transform output agree within tolerance

#### Scenario: In-memory intent collects

- **WHEN** the execution intent is explicitly in-process and the source is streamed
- **THEN** the batches are collected and the dense path produces the model

#### Scenario: Empty source

- **WHEN** `fit_streaming` receives a source yielding no batch
- **THEN** it returns a structured error instead of a vacuous model

### Requirement: The configured backend is carried and recorded

Per the `backend-selection` capability, every estimator operation SHALL record the
configured `SKBackendKind` in its observability span — never a hardcoded value — and `fit`
SHALL transfer the kind into the fitted model, whose operations record the carried kind.

#### Scenario: Model inherits the estimator backend

- **WHEN** an estimator built with a non-default backend is fitted
- **THEN** the fitted model reports and records that same backend kind
