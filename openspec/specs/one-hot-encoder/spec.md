# one-hot-encoder Specification

## Purpose
Defines `SKOneHotEncoder`, the Wave 1 categorical transformer: a builder-configured,
backend-carrying encoder that turns categorical features — borrowed text labels or small
integer codes — into a numeric indicator representation, emitted either densely or as a
`sprs` CSR matrix, with a deterministic per-column label table as its exportable state.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKOneHotEncoder` SHALL be constructed only through `SKOneHotEncoder::build()`, which SHALL
expose chainable `sparse_output(bool)` (defaulting to `true`), `handle_unknown(SKUnknownPolicy)`
(defaulting to `Reject`), `backend(SKBackendKind)` (defaulting to `SKBackendKind::Faer`) and
`execution_mode(SKExecutionMode)` (defaulting to `Automatic`); the direct constructor SHALL
stay private.

#### Scenario: Default configuration

- **WHEN** an encoder is built without calling any setter
- **THEN** it requests sparse output, rejects unknown categories, carries
  `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Configuration override flows through the model

- **WHEN** an encoder is built with `sparse_output(false)` and a non-default backend
- **THEN** the fitted model emits dense output and reports the carried backend kind

### Requirement: Categorical input accepts text and integer codes

`fit` SHALL accept a categorical feature view carrying either a nominal (text) matrix or
integer-coded columns; its learned tables are independent of the output scalar. The transforms
SHALL be generic over the output scalar `F: SKFloat`. Integer columns SHALL treat a documented
sentinel code as a missing entry that is not registered as a category.

#### Scenario: Text columns

- **WHEN** a nominal matrix of two text columns is fitted
- **THEN** each column gets its own label table over the distinct labels it observed

#### Scenario: Integer-coded columns

- **WHEN** an integer-coded matrix is fitted
- **THEN** the observed codes become the column's categories in first-occurrence order

#### Scenario: Both supported output floats

- **WHEN** the same categorical data is fitted for `f32` and for `f64` output
- **THEN** both fits succeed and produce equal-shaped output in their own scalar

### Requirement: Deterministic per-column category tables

The fitted model SHALL expose one reversible category table per input column, assigning
category indices in first-occurrence order so that the same input always yields the same
tables: a label table for nominal columns and a code table for integer-coded columns, each
mapping a category to its index and back. The total encoded width SHALL be the sum of the
columns' category counts.

#### Scenario: Same input yields the same tables

- **WHEN** the same categorical data is fitted twice
- **THEN** both models expose identical category tables

#### Scenario: Roundtrip through a nominal table

- **WHEN** a label is looked up to an index and the index back to a label
- **THEN** the original label is recovered

#### Scenario: Roundtrip through an integer table

- **WHEN** a code is looked up to an index and the index back to a code
- **THEN** the original code is recovered

### Requirement: Dense transform produces indicator columns

`transform_dense` SHALL return an owned dense matrix with one column per known category,
placing a unit in the column matching each row's category and the additive identity elsewhere,
using only the model's learned tables. A column-count or category mismatch SHALL be resolved
per the `handle_unknown` policy before any element is written.

#### Scenario: Hand-checkable encoding

- **WHEN** a single text column `["a", "b", "a"]` is fitted and transformed densely
- **THEN** the output is `[[1, 0], [0, 1], [1, 0]]` with categories ordered `a`, `b`

#### Scenario: Output column count

- **WHEN** several columns with distinct category counts are encoded
- **THEN** the output width equals the sum of the category counts

### Requirement: Sparse transform equals the dense transform

`transform_sparse` SHALL return a `sprs` CSR matrix whose nonzero structure matches the dense
transform exactly (one stored unit per row per encoded value, sum of stored values per row
equal to the number of well-defined input columns); `sparse_output(true)` SHALL make the
default transform path return the sparse form and `sparse_output(false)` the dense form.

#### Scenario: Sparse matches dense

- **WHEN** the same data is transformed densely and sparsely
- **THEN** densifying the sparse result equals the dense result exactly

#### Scenario: Builder selects the default output

- **WHEN** two encoders differing only in `sparse_output` transform the same data
- **THEN** each returns its configured representation

### Requirement: Unknown categories honor the policy

`handle_unknown` SHALL govern categories not present in a column's table: `Reject` SHALL fail
with a structured error before writing output, and `Ignore` SHALL encode the entry as all
zeros for that column without failing.

#### Scenario: Reject by default

- **WHEN** the default encoder transforms a category it did not learn
- **THEN** it returns a structured error and produces no output

#### Scenario: Ignore encodes zeros

- **WHEN** an encoder built with `handle_unknown(Ignore)` transforms an unknown category
- **THEN** that column's block is all zeros and the operation succeeds

### Requirement: Missing codes are not categories

An integer-coded entry equal to the documented missing sentinel SHALL NOT be registered as a
category during fit and SHALL encode as all zeros (or under the unknown policy) during
transform, consistently across the dense and sparse paths.

#### Scenario: Sentinel excluded from the table

- **WHEN** a column contains the missing sentinel alongside real codes
- **THEN** the sentinel does not appear in the column's label table

#### Scenario: Sentinel encodes to nothing

- **WHEN** a row carries the sentinel in a column
- **THEN** that column's block is all zeros

### Requirement: Out-of-core fit_streaming equals the dense fit

`fit_streaming` SHALL consume a sequence of categorical batches and produce the same tables the
dense `fit` would produce for the concatenated batches: category indices follow first-occurrence
order across the concatenation. An empty sequence or a batch sequence with inconsistent column
counts SHALL fail with a structured error. Because the tables depend only on the categories
observed in order, accumulating them batch by batch is exact — no separate in-memory branch is
required.

#### Scenario: Uneven batches equal dense

- **WHEN** the same data is streamed in uneven batches and compared with the dense fit
- **THEN** the label tables and the transform output agree exactly

#### Scenario: Empty sequence

- **WHEN** `fit_streaming` receives no batch
- **THEN** it returns a structured error instead of a vacuous model

### Requirement: Structured errors on invalid input

The encoder SHALL reject empty categorical inputs and inconsistent column counts with the
central `SKError` taxonomy (`ShapeMismatch`, `UnsupportedRepresentation`), before processing
any element.

#### Scenario: Empty input

- **WHEN** `fit` receives a categorical matrix with zero rows or zero columns
- **THEN** it returns a `ShapeMismatch` error before computing anything

#### Scenario: Inconsistent batches

- **WHEN** `fit_streaming` receives a batch whose column count differs from the first
- **THEN** it returns a `ShapeMismatch` error and the model is unchanged

### Requirement: The configured backend is carried and recorded

Per the `backend-selection` capability, every encoder operation SHALL record the configured
`SKBackendKind` in its observability span — never a hardcoded value — and `fit` SHALL transfer
the kind into the fitted model, whose operations record the carried kind.

#### Scenario: Model inherits the estimator backend

- **WHEN** an encoder built with a non-default backend is fitted
- **THEN** the fitted model reports and records that same backend kind
