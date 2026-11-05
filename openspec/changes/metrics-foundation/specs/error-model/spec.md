# Spec Delta

## ADDED Requirements

### Requirement: Empty observation set error

The central taxonomy SHALL represent an operation that requires at least one observation
but received none, naming the operation, and SHALL NOT return a numeric sentinel or panic.

#### Scenario: Empty evaluation is rejected

- **WHEN** a metric is computed over zero observations
- **THEN** the returned error names the operation and reports that observations are required

### Requirement: Non-finite numeric value error

The central taxonomy SHALL represent a non-finite numeric value received where finite
values are required, carrying the zero-based position of the offending value.

#### Scenario: NaN identifies its position

- **WHEN** a prediction at position three is NaN
- **THEN** the returned error reports `NonFiniteValue` with position three

### Requirement: Unknown label error

The central taxonomy SHALL represent a label observed outside an explicitly supplied label
space, carrying the offending label value.

#### Scenario: Unknown label carries its value

- **WHEN** an observation carries label `7` and the explicit label space is `[0, 1]`
- **THEN** the returned error reports `UnknownLabel` with label `7`
