# missing-indicator Specification

## Purpose
The missingness mask: which feature columns the mask covers after fit, how the mask is
represented and returned for dense or sparse input, and how transform treats missingness
that fit never saw.

## Requirements

### Requirement: Builder-only construction with documented defaults

`SKMissingIndicator` SHALL be constructible only through its builder, exposing chainable
`features(SKMissingFeatureScope)` defaulting to `MissingOnly`, `error_on_new(bool)`
defaulting to `true`, `sparse_output(bool)` defaulting to `false`,
`backend(SKBackendKind)` defaulting to `Faer` and `execution_mode(SKExecutionMode)`
defaulting to `Automatic`; the direct constructor SHALL stay private.

#### Scenario: Default configuration

- **WHEN** an indicator is built without calling any setter
- **THEN** it masks only the features that were missing at fit time, rejects new missingness at transform, emits a dense mask, and carries `Faer` with an `Automatic` intent

#### Scenario: Masking every feature

- **WHEN** a builder sets `features(All)`
- **THEN** the fitted model emits one mask column per input feature, missing or not

### Requirement: Fit learns the feature scope

Fit SHALL record, from the fitted matrix, the set of feature columns that contained at
least one missing cell, together with the total column count and the carried backend kind.
Under `MissingOnly` that set is the mask scope; under `All` the scope is every column.

#### Scenario: Missing-only scope

- **WHEN** a three-column matrix missing only in its second column is fitted under `MissingOnly`
- **THEN** the model's scope is exactly the second column

#### Scenario: Deterministic scope across fits

- **WHEN** the same estimator fits the same matrix twice through a shared reference
- **THEN** both models expose the same scope and column count

### Requirement: Transform returns the fitted-scope mask in the scalar type

Transform SHALL return a `0.0`/`1.0` matrix in the scalar type — dense for dense input,
CSR for sparse input — with one column per feature in the fitted scope, marking exactly the
cells that are missing under the marker semantics. An input whose column count differs from
the fitted count SHALL be rejected with the standard shape mismatch error.

#### Scenario: Mask marks the missing cells

- **WHEN** a model fitted under `All` transforms `[[NaN, 1.0], [2.0, NaN]]`
- **THEN** the mask is `[[1.0, 0.0], [0.0, 1.0]]`

#### Scenario: Sparse input yields a sparse mask

- **WHEN** a CSR matrix with an explicit missing entry is transformed under a sparse input
- **THEN** the mask is a CSR matrix whose stored entries are exactly the missing cells

### Requirement: New missingness is rejected when error_on_new is set

Under `MissingOnly`, transform SHALL fail with a precise error when the input carries a
missing cell in a feature that was complete at fit time, while `error_on_new(false)` lets
that cell pass unmarked. The check SHALL apply to every cell of the offending feature.

#### Scenario: New missingness fails fast

- **WHEN** a model fitted on data complete in the first column transforms a matrix missing in that column with the default configuration
- **THEN** the operation fails with an error naming the never-missing feature

#### Scenario: New missingness tolerated on request

- **WHEN** the same transform runs with `error_on_new(false)`
- **THEN** the call succeeds and the new missingness stays unmarked
