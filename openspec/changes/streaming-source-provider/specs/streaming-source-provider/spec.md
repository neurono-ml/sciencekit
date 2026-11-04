## Purpose

Defines the source-provider seam shared by every out-of-core fit: a supplier that opens one
streaming pass at a time, satisfied both by a concrete sequential source and by a reopenable
factory, with one rule for how many passes each estimator class may take.

## ADDED Requirements

### Requirement: A provider opens one pass at a time

The library SHALL define a supervised source provider and a feature-only source provider.
Each SHALL open one source per call, SHALL declare whether it can be reopened, and SHALL be
usable without dynamic dispatch. Opening SHALL be fallible and SHALL report errors through
the central taxonomy.

#### Scenario: A provider opens a source

- **WHEN** a provider is asked for a pass
- **THEN** it returns a source that yields batches until its end

#### Scenario: Reopenability is declared

- **WHEN** a provider is constructed
- **THEN** it reports whether further passes are possible

### Requirement: A concrete source is a one-pass provider

The library SHALL implement the provider contract for a mutable reference to any sequential
source, so an existing call that passes a source keeps compiling. That provider SHALL report
itself as not reopenable and SHALL yield the same source once.

#### Scenario: Existing call shape is preserved

- **WHEN** an estimator's streaming fit receives a mutable reference to a source
- **THEN** the call compiles and consumes the source

#### Scenario: A concrete source reports no reopen

- **WHEN** the provider wrapping a concrete source is asked whether it can reopen
- **THEN** it reports that no further pass is available

### Requirement: A reopenable provider yields a fresh source per pass

The library SHALL provide a reopenable provider wrapping a fallible factory. Each pass SHALL
obtain a source from the factory, so the estimator observes the same data in the same order
every epoch. A factory error SHALL abort the fit with the central error carrying the original
cause.

#### Scenario: Two passes read the data twice

- **WHEN** a reopenable provider is used with a budget of two passes
- **THEN** the factory is invoked twice and both passes see the full data

#### Scenario: A factory failure aborts the fit

- **WHEN** the factory fails on the second pass
- **THEN** the fit returns the wrapped error and does not report a model

### Requirement: The number of passes follows the estimator class

An iterative estimator SHALL take up to `maximum_number_of_iterations` passes from a
reopenable provider. A non-iterative estimator whose statistics cannot be re-fed SHALL take
exactly one pass. When a one-shot provider is used with a budget above one, the executor
SHALL take one pass and emit a warning, and the fitted model SHALL report the number of
passes actually taken.

#### Scenario: An iterative estimator honours the budget

- **WHEN** an iterative estimator is streamed from a reopenable provider with a budget of
  several passes
- **THEN** it takes those passes and reports them

#### Scenario: A one-shot provider is consumed once

- **WHEN** an estimator is streamed from a concrete source with a budget above one
- **THEN** exactly one pass is taken, a warning is emitted and the model reports one pass

#### Scenario: A pass-limited fit is not a convergence failure

- **WHEN** an iterative estimator under a strict convergence policy is limited to fewer
  passes than its budget
- **THEN** it returns its model reporting the passes taken and that it did not converge,
  instead of a non-convergence error

#### Scenario: An accumulating estimator takes one pass

- **WHEN** an estimator that accumulates exact statistics is streamed from a reopenable
  provider with a budget above one
- **THEN** it takes exactly one pass and its statistics are not double counted

### Requirement: Unsatisfiable streaming requests are refused

A streaming request the provider cannot satisfy SHALL return
`SKError::UnsupportedOperation { operation, reason }`: early stopping over a stream, or any
request that needs a validation slice the provider cannot supply. Silently degrading SHALL
not happen.

#### Scenario: Early stopping over a stream is refused

- **WHEN** a streaming fit is asked to early-stop on held-out data
- **THEN** it returns an `UnsupportedOperation` error naming the operation and the reason

### Requirement: Both provider families behave identically

The feature-only and supervised providers SHALL expose the same opening, reopenability and
error semantics; the supervised family SHALL additionally carry the targets paired with each
batch.

#### Scenario: Feature-only and supervised providers agree

- **WHEN** the same provider logic is exercised over features only and over features with
  targets
- **THEN** the number of passes, the order of the batches and the error behaviour are the
  same
