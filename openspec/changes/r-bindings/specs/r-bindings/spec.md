# Spec Delta

## Purpose

Defines the R package `sciencekitr`: native routine registration, external-pointer
lifecycle, dense matrix conversion, operations, structured errors and packaging.

## ADDED Requirements

### Requirement: Module loading and native routine registration

The R package SHALL load its native library and expose the documented constructors, and
SHALL register its native routines with `R_registerRoutines` so that they are reachable
only through registered `.Call` entry points.

#### Scenario: Loading the package

- **WHEN** R loads the `sciencekitr` package
- **THEN** the package loads without error and the documented constructors are exported

#### Scenario: Registered routines only

- **WHEN** the package is loaded
- **THEN** dynamic symbol lookup is disabled and the native routines are reachable only
  through their registered names

### Requirement: Builder and model handles

Builders and fitted models SHALL be represented as R external pointers that own the
native handle, with a finalizer that releases it and an explicit `close()` that performs
the same release eagerly.

#### Scenario: Explicit close

- **WHEN** `close()` is called on a builder or a model
- **THEN** the native handle is released exactly once

#### Scenario: Idempotent close

- **WHEN** `close()` is called twice on the same object
- **THEN** the second call is a no-op and raises no error and no double free occurs

#### Scenario: Operation on a closed handle

- **WHEN** an operation is invoked on an object whose handle was released
- **THEN** a structured error with the `invalid_handle` status is raised and no freed
  memory is dereferenced

#### Scenario: Finalizer

- **WHEN** a model becomes unreachable and the garbage collector runs
- **THEN** its native handle is freed exactly once

### Requirement: Dense matrix conversion

Dense input SHALL be converted to the ABI's array descriptor with explicit rectangularity
and type validation, and the binding SHALL support `f64` semantics only.

#### Scenario: Numeric matrix

- **WHEN** a numeric matrix of doubles is passed to `fit`, `predict` or `transform`
- **THEN** the native call reads the values without corruption and the result has the
  expected shape

#### Scenario: Non-rectangular input

- **WHEN** a data frame with ragged columns is passed as the design matrix
- **THEN** the binding raises a shape error before calling the native library

#### Scenario: Unsupported element type

- **WHEN** an integer, logical or character matrix is passed
- **THEN** the binding raises a condition explaining that only double matrices are
  supported, instead of silently converting

#### Scenario: Shape mismatch

- **WHEN** the number of rows of the features and the targets differ
- **THEN** the binding raises a shape error before calling the native library

### Requirement: Operations

The package SHALL expose `fit`, `predict`, `transform`, `score`, `save` and `load` over the
builder and model objects, for the estimator families the ABI supports.

#### Scenario: Fit and predict

- **WHEN** a linear-regression builder is configured and fitted on a known fixture
- **THEN** `predict` returns the expected values for the fixture

#### Scenario: Transform

- **WHEN** a fitted standard scaler transforms a matrix
- **THEN** the transformed matrix has the expected shape and values

#### Scenario: Score

- **WHEN** a fitted model is scored against known targets
- **THEN** the score is the value the native library reports

#### Scenario: Persistence round-trip

- **WHEN** a fitted model is saved and loaded again
- **THEN** the loaded model predicts the same values as the saved one

### Requirement: Structured errors

ABI failures SHALL become R conditions of class `sciencekit_error` carrying the ABI status
name, the numeric status code, the failing operation and the native message.

#### Scenario: Invalid input

- **WHEN** a native call fails
- **THEN** R raises a `sciencekit_error` condition whose `status` field is the ABI status
  name and whose `status_code` field is the numeric ABI status

#### Scenario: Native message preserved

- **WHEN** the native library records a diagnostic message
- **THEN** the raised condition carries that message in its `message` field

### Requirement: Lifecycle safety

Repeated failures SHALL NOT leak native resources, and the binding SHALL NOT use
released memory.

#### Scenario: Repeated failures

- **WHEN** the same failing operation is invoked many times in a row
- **THEN** the number of live native handles does not grow and the process stays stable

### Requirement: Packaging and tests

The package SHALL ship `DESCRIPTION`, `NAMESPACE`, `R/`, `src/`, a `testthat` test suite
and a CI workflow that builds the native library and runs `R CMD check .`.

#### Scenario: Test suite

- **WHEN** the testthat suite runs
- **THEN** the module load, fit/predict/score, transform, persistence, error mapping, close
  idempotency and no-leak fixtures all pass

#### Scenario: Package check

- **WHEN** `R CMD build .` followed by `R CMD check .` runs
- **THEN** the check completes without errors
