# polynomial-features Specification

## Purpose

Defines `SKPolynomialFeatures`, the Wave 1 numeric expander: a builder-configured,
backend-carrying transformer that turns a feature matrix into its polynomial and interaction
terms, emitted either densely or as a `sprs` CSR matrix, driven by a lazy exponent generator
so the combination set is never materialized as a blow-up of owned arrays.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKPolynomialFeatures` SHALL be constructed only through `SKPolynomialFeatures::build()`, which
SHALL expose chainable `degree(usize)` (defaulting to `2`), `interaction_only(bool)`
(defaulting to `false`), `include_bias(bool)` (defaulting to `true`), `sparse_output(bool)`
(defaulting to `false`), `backend(SKBackendKind)` (defaulting to `SKBackendKind::Faer`) and
`execution_mode(SKExecutionMode)` (defaulting to `Automatic`); the direct constructor SHALL
stay private.

#### Scenario: Default configuration

- **WHEN** an expander is built without calling any setter
- **THEN** it requests degree two, includes interaction *and* power terms, includes a bias
  column, emits dense output, carries `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Invalid degree is rejected

- **WHEN** an expander is built with `degree(0)`
- **THEN** construction returns an `InvalidHyperparameter` error

### Requirement: Fit learns the expansion shape

`fit` SHALL accept a numeric feature view, learn the input column count, and return a fitted
model exposing the input feature count and the total output feature count derived from
`degree`, `interaction_only` and `include_bias`. The fitted state is independent of the output
scalar; the transforms SHALL be generic over the scalar `F: SKFloat`.

#### Scenario: Hand-checkable two-feature degree two

- **WHEN** a two-column matrix is fitted with defaults
- **THEN** the model reports six output features for the terms `1`, `x0`, `x1`, `x0²`, `x0x1`,
  `x1²`

#### Scenario: Interaction only

- **WHEN** `interaction_only(true)` is set on two features at degree two
- **THEN** the pure power terms `x0²` and `x1²` are absent from the expansion

#### Scenario: Bias excluded

- **WHEN** `include_bias(false)` is set
- **THEN** the constant term is absent and the output width drops by one

### Requirement: Stable, documented column order

The expansion SHALL follow a documented, deterministic column order — the constant bias column
first (when included), then the terms in increasing total degree, and within a degree in
lexicographic order of exponents over the input columns — so the same configuration always
produces the same column layout.

#### Scenario: Deterministic layout

- **WHEN** the same data is expanded twice with the same configuration
- **THEN** the column layout is identical

#### Scenario: Degree ordering

- **WHEN** a degree-two expansion is produced
- **THEN** lower-degree terms precede higher-degree terms, with the bias term first

### Requirement: Dense transform expands the features

`transform_dense` SHALL return an owned dense matrix holding each row's polynomial terms in
the model's column order, computed from the input row alone; a column-count mismatch SHALL
fail with a `ShapeMismatch` error before any element is written.

#### Scenario: Hand-checkable expansion

- **WHEN** the row `[2, 3]` is transformed with defaults
- **THEN** the output row is `[1, 2, 3, 4, 6, 9]`

#### Scenario: Column mismatch

- **WHEN** a fitted model transforms data whose column count differs from its own
- **THEN** it returns a `ShapeMismatch` error and produces no output

### Requirement: Sparse transform equals the dense transform

`transform_sparse` SHALL return a `sprs` CSR matrix whose nonzero structure derives exactly from
the dense expansion (only zeros are omitted); `sparse_output(true)` SHALL make the default
transform path return the sparse form and `sparse_output(false)` the dense form.

#### Scenario: Sparse matches dense

- **WHEN** the same data is expanded densely and sparsely
- **THEN** densifying the sparse result equals the dense result exactly

#### Scenario: Zeros are not stored

- **WHEN** a term evaluates to zero for a row
- **THEN** no entry is stored for that row and column

### Requirement: Out-of-core fit_streaming equals the dense fit

`fit_streaming` SHALL consume a sequential `SKLazySource` and produce the same model the dense
`fit` would produce for the concatenated data: under an in-memory execution intent it SHALL
collect the batches and reuse the dense path; otherwise it SHALL learn the feature count from
the first non-empty batch. An empty source or a batch sequence with inconsistent column counts
SHALL fail with a structured error.

#### Scenario: Uneven batches equal dense

- **WHEN** the data is streamed in uneven batches and compared with the dense fit
- **THEN** the input feature count and the transform output agree within tolerance

#### Scenario: Empty source

- **WHEN** `fit_streaming` receives a source yielding no batch
- **THEN** it returns a structured error instead of a vacuous model

### Requirement: Structured errors on invalid input

The expander SHALL reject empty inputs, zero features and inconsistent column counts with the
central `SKError` taxonomy (`ShapeMismatch`, `InvalidHyperparameter`), before processing any
element, and SHALL reject sparse *input* at the dense-only boundary with
`UnsupportedRepresentation`.

#### Scenario: Empty input

- **WHEN** `fit` receives a matrix with zero rows or zero columns
- **THEN** it returns a `ShapeMismatch` error before computing anything

#### Scenario: Sparse input rejected

- **WHEN** `fit` receives a sparse matrix
- **THEN** it returns an `UnsupportedRepresentation` error at the dense-input boundary

### Requirement: The configured backend is carried and recorded

Per the `backend-selection` capability, every expander operation SHALL record the configured
`SKBackendKind` in its observability span — never a hardcoded value — and `fit` SHALL transfer
the kind into the fitted model, whose operations record the carried kind.

#### Scenario: Model inherits the estimator backend

- **WHEN** an expander built with a non-default backend is fitted
- **THEN** the fitted model reports and records that same backend kind
