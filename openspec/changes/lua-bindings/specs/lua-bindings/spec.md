# Spec Delta

## Purpose

Defines the Lua module: loading, userdata lifecycle, table/array conversion, operations,
errors and packaging.

## ADDED Requirements

### Requirement: Module loading

The Lua package SHALL expose a `sciencekit` module that loads the native library and
provides the documented constructors.

#### Scenario: Require

- **WHEN** Lua calls `require("sciencekit")`
- **THEN** the module loads and exposes the builder constructors

### Requirement: Builders and models

Builders and fitted models SHALL be Lua userdata with chainable setters and the supported
operations.

#### Scenario: Fit and predict

- **WHEN** a Lua builder is configured, fitted and used to predict
- **THEN** predictions are returned as a table of rows

#### Scenario: Transform

- **WHEN** a transformer model transforms a table
- **THEN** the transformed table has the expected shape

### Requirement: Array conversion

Lua tables SHALL convert to/from numeric arrays with explicit shape validation; copy
semantics SHALL be documented.

#### Scenario: Ragged table

- **WHEN** a table has rows of different lengths
- **THEN** the binding raises a shape error before calling the library

### Requirement: Errors and lifecycle

Library errors SHALL become Lua errors with status/message, and userdata `__gc` SHALL free
the underlying handle exactly once.

#### Scenario: Invalid input

- **WHEN** an operation receives an invalid array
- **THEN** Lua receives an error with the structured message

#### Scenario: Garbage collection

- **WHEN** a model userdata becomes unreachable
- **THEN** its handle is freed and no double free occurs

### Requirement: Packaging

The package SHALL ship a rockspec and a Lua 5.4 test suite.

#### Scenario: Lua test suite

- **WHEN** the Lua tests run
- **THEN** every family's fit/predict fixture passes
