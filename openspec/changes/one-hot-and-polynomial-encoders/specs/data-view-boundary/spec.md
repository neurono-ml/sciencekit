# Spec Delta

## ADDED Requirements

### Requirement: Categorical feature view as an extensible enum

The library SHALL expose a categorical feature view carrying the representations that are not
numeric feature vectors — nominal (text) columns and integer-coded columns — marked as
non-exhaustive so new categorical representations can be added without breaking consumers. It
SHALL be the categorical counterpart of the numeric feature view, borrowing its data
zero-copy.

#### Scenario: Nominal matrix enters without copying

- **WHEN** a row-major matrix of borrowed text labels is passed to an operation as a
  categorical view
- **THEN** the conversion produces the nominal variant referencing the same text without
  copying it

#### Scenario: Integer-coded matrix enters through the same boundary

- **WHEN** a matrix of category codes is passed to an operation as a categorical view
- **THEN** the conversion produces the integer variant referencing the same data

### Requirement: Universal conversion seam at categorical inputs

Every public categorical input SHALL be declared over the standard library's fallible
conversion trait, so that types with infallible conversion are accepted automatically and
external integrators can plug their own types — including fallible ones — into the categorical
view; a failed conversion SHALL reach the consumer through the operation's `Result`.

#### Scenario: Infallible conversion accepted automatically

- **WHEN** a consumer passes a type implementing only infallible conversion into the
  categorical view
- **THEN** the call compiles and runs because the std promotes that conversion to the required
  fallible form

#### Scenario: Fallible conversion reports a structured error

- **WHEN** an external conversion into the categorical view fails during a call
- **THEN** the error reaches the consumer in the conversion-failure variant, without panicking

### Requirement: Precise rejection of unsupported categorical representation

Algorithms supporting only part of the categorical representations SHALL reject the others at
runtime with a specific unsupported-representation error, dispatching on representation exactly
once per operation, never per element.

#### Scenario: Unsupported variant is rejected once

- **WHEN** a categorical representation an algorithm does not support is delivered to it
- **THEN** the operation fails before processing any element, with an error naming the
  representation and a conversion path

### Requirement: Documented missing-code sentinel

The integer-coded representation SHALL document a single sentinel code denoting a missing
categorical entry, distinct from every valid category code, so consumers can agree on missing
handling across encoders and imputers.

#### Scenario: Sentinel is documented and distinct

- **WHEN** a consumer reads the integer-coded representation contract
- **THEN** the missing sentinel value is specified and does not collide with a valid category
  code by convention
