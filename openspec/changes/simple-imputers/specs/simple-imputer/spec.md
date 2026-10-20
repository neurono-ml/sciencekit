# Spec Delta

## Purpose

Univariate missing-value imputation: which cells count as missing per representation, which
per-column statistic fills them under each strategy, how empty and incremental data behave,
and how the imputed output preserves the input's dense or sparse shape.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKSimpleImputer` SHALL be constructible only through its builder, exposing chainable
`strategy(SKImputationStrategy)` defaulting to `Mean`, `missing_values(F)` defaulting to
`NaN`, `fill_value(Option<F>)`, `keep_empty_features(bool)` defaulting to `false` and
`add_indicator(bool)` defaulting to `false`; the direct constructor SHALL stay private.

#### Scenario: Default configuration

- **WHEN** an imputer is built without calling any setter
- **THEN** it fills with the column mean, treats `NaN` as missing, drops empty columns and stacks no indicator

#### Scenario: Strategy and marker overrides

- **WHEN** a builder sets `strategy(Constant)` with `fill_value(Some(1.5))` and `missing_values(0.0)`
- **THEN** the fitted model fills every `0.0` cell with `1.5`, whatever the column contents

### Requirement: Generic over the scalar, never over the backend

The estimator SHALL stay generic over the scalar `SKFloat` — its configuration carries `F`
values — and SHALL keep the backend a runtime value, never a type parameter (spec
`backend-selection`).

#### Scenario: One estimator type serves both scalars

- **WHEN** the same imputer configuration is built and fitted for `f32` and for `f64`
- **THEN** a single estimator type serves both, no backend type parameter appears, and each fit yields a model of the matching scalar

### Requirement: Backend and execution intent selection

The builder SHALL expose `backend(SKBackendKind)` defaulting to `Faer` and
`execution_mode(SKExecutionMode)` defaulting to `Automatic`. Every operation the estimator
or its model runs SHALL record the configured kind in its span, and `fit` SHALL transfer the
kind into the fitted model, whose operations then record it.

#### Scenario: Default kind reaches the model

- **WHEN** an imputer is fitted with the default configuration
- **THEN** the fitted model records `Faer` in every operation span it emits

#### Scenario: Explicit backend override

- **WHEN** a builder sets `backend(...)` to a non-default kind and fits
- **THEN** that configured kind, never a hardcoded one, is the value recorded by fit and transform

### Requirement: Dense fit learns per-column strategy statistics

Dense `fit` SHALL compute, for each column over its non-missing cells, the strategy
statistic: exact mean (sum/count), exact median (order statistic of the sorted values),
most frequent value with ties broken toward the smallest value, or the configured constant.
The returned model SHALL own those statistics, the set of columns that produced no
statistic, and the carried backend kind.

#### Scenario: Mean over a hand-checkable column

- **WHEN** a column `[7.0, NaN, 10.0]` is fitted with the default strategy
- **THEN** its statistic is the mean of the observed values, `8.5`

#### Scenario: Median over an even observed count

- **WHEN** a column `[1.0, NaN, 3.0, 5.0, 7.0]` is fitted with `Median`
- **THEN** its four observed values average their central pair, so the statistic is `4.0`

#### Scenario: Most-frequent tie breaks toward the smallest value

- **WHEN** a column `[2.0, 5.0, 2.0, 5.0, NaN]` is fitted with `MostFrequent`
- **THEN** its statistic is `2.0`, the smallest of the tied modes

### Requirement: Sparse marker semantics

For a CSR input, an explicit entry SHALL be missing exactly when it equals the configured
marker. When the marker is `0.0`, the implicit zeros SHALL also be missing; for any other
marker the implicit zeros SHALL count as observed values. Statistics SHALL be computed over
that same observed set.

#### Scenario: Explicit NaN in a CSR matrix

- **WHEN** a CSR column stores `[1.0, NaN]` explicitly with the default marker
- **THEN** the statistic ignores the stored `NaN` and the implicit zeros are observed values

#### Scenario: Zero marker treats implicit zeros as missing

- **WHEN** a CSR column stores no explicit entries at all with `missing_values(0.0)`
- **THEN** every cell of the column is missing, so the column is empty under a non-constant strategy

### Requirement: Transform preserves representation and fills only missing cells

`transform` SHALL return an owned `SKImputedOutput<F>` that is dense for dense input and
CSR for sparse input, SHALL replace exactly the missing cells with the fitted statistic (or
the retained empty-column value) and SHALL leave every observed cell untouched. An input
whose column count differs from the fitted count SHALL be rejected with the standard shape
mismatch error before any element is read.

#### Scenario: Only missing cells change

- **WHEN** the model transforms `[NaN, 2.0, 4.0]` fitted with mean
- **THEN** the output is `[3.0, 2.0, 4.0]`, with the observed cells bit-identical to the input

#### Scenario: Column-count mismatch is rejected

- **WHEN** a model fitted on three columns transforms a two-column matrix
- **THEN** the operation fails with the standard shape mismatch error and returns no output

#### Scenario: Sparse input yields sparse output

- **WHEN** the model transforms a CSR matrix fitted from CSR data
- **THEN** the output is a CSR matrix of the same shape with only the filled cells added as stored entries

### Requirement: All-missing column policy

A column with no observed value SHALL receive a `NaN` statistic and be dropped from the
transformed output when the strategy is not `Constant`. It SHALL be retained and filled
with `0.0` instead when `keep_empty_features(true)` is set, or filled with `fill_value`
whenever the strategy is `Constant`.

#### Scenario: Empty column dropped by default

- **WHEN** a column is missing in every fitted row and the strategy is `Mean`
- **THEN** the transformed matrix has one column fewer and the model reports that index as dropped

#### Scenario: Empty column retained on request

- **WHEN** the same fit runs with `keep_empty_features(true)`
- **THEN** the column is present in the output and every cell of it is `0.0`

### Requirement: Incremental fit matches the concatenated fit

`partial_fit` SHALL accumulate the same state that `fit` derives from the concatenated
batches — exact for `Mean`, `MostFrequent` and `Constant` — so a model fitted batch by batch
is equivalent to one fitted on the whole at once. An empty batch SHALL be a no-op, and a
batch whose column count differs from the accumulated one SHALL be rejected.

#### Scenario: Split batches equal the whole

- **WHEN** a model is built with `partial_fit` over two uneven batches and compared with `fit` on their concatenation
- **THEN** the statistics are equal and both models transform a probe matrix identically

#### Scenario: Split batches extend the indicator scope

- **WHEN** a model built with `add_indicator(true)` is fitted on a first batch and extended with a second batch that introduces a missing column
- **THEN** its indicator scope and stacked output equal those of the model fitted on the concatenation

#### Scenario: Empty batch is a no-op

- **WHEN** `partial_fit` receives a batch with zero rows
- **THEN** the accumulated statistics are unchanged and the call succeeds

### Requirement: Streaming fit honors the execution intent

`fit_streaming` SHALL produce the same model the dense `fit` would produce for the
concatenated data when the resolved execution intent is in-process, and SHALL otherwise run
through the streaming driver with per-batch accumulators — exact for mean, mode and
constant, a `tdigest` sketch for the median. Every batch SHALL be instrumented as a fit
operation whose span records the configured backend kind.

#### Scenario: In-memory intent matches the dense fit

- **WHEN** `fit_streaming` runs under an in-process synchronous intent over batches
- **THEN** the resulting statistics equal the dense `fit` on the concatenation

#### Scenario: Driver median stays within sketch tolerance

- **WHEN** `fit_streaming` runs through the streaming driver with `Median`
- **THEN** the fitted median lies within the sketch's documented tolerance of the exact median

### Requirement: Indicator columns stack onto the imputed output

When `add_indicator(true)` is set, `transform` SHALL horizontally stack one `0.0`/`1.0`
column per feature that was missing at fit time onto the imputed matrix, ordered after the
retained feature columns, so the output width equals the retained feature count plus the
counted missing features. The model SHALL expose the fitted `SKMissingIndicator` it uses.

#### Scenario: Stacked width

- **WHEN** an imputer fitted on three columns with one missing feature transforms a matrix with `add_indicator(true)`
- **THEN** the output has four columns: three imputed plus the indicator of the missing feature
