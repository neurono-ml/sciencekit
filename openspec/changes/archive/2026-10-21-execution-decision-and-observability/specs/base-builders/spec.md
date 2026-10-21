## Purpose

The mandatory builder pattern foundation: every public estimator, transformer and algorithm
exposes a typed builder with an execution-intent setter defaulting to `Automatic` and a
`build()` method returning a `Result`, so consumers configure and construct models
consistently across the library.

## ADDED Requirements

### Requirement: Every estimator exposes a builder
The library SHALL expose every public estimator, transformer and algorithm through a typed
builder whose direct constructor is private, so construction always goes through `build()`.

#### Scenario: A model is built through its builder
- **WHEN** a consumer configures a builder and calls `build()`
- **THEN** it obtains the constructed estimator, and no direct constructor is available

### Requirement: Execution mode defaults to Automatic
Every builder SHALL expose an `execution_mode(...)` setter accepting an `SKExecutionMode`
whose default is `Automatic`, so a consumer who does not set it still gets automatic
execution.

#### Scenario: Default build uses automatic execution
- **WHEN** a builder is used without setting the execution mode
- **THEN** the constructed estimator carries the `Automatic` execution intent

#### Scenario: Explicit mode overrides the default
- **WHEN** a consumer sets a specific execution mode on the builder
- **THEN** the constructed estimator carries that explicit intent

### Requirement: Invalid builder state fails at build time
The library SHALL return a `Result` from `build()`, reporting invalid hyperparameter values
or incompatible configuration through the central error taxonomy rather than panicking.

#### Scenario: An invalid hyperparameter surfaces as an error
- **WHEN** a builder is configured with an invalid hyperparameter value and built
- **THEN** `build()` returns an error identifying the offending hyperparameter