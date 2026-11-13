# json-debug-format Specification

## Purpose
Defines the human-readable JSON debug representation of a model state and its import.

## Requirements

### Requirement: JSON schema

The debug format SHALL contain a format version, estimator metadata, typed hyperparameters
and a named tensor map with dtype, shape and values.

#### Scenario: Metadata is readable

- **WHEN** a model state is written as JSON
- **THEN** the estimator name, scalar dtype, feature count and hyperparameters are
  human-readable fields

#### Scenario: Tensor values are readable

- **WHEN** a tensor is written
- **THEN** its shape and flat values are visible without binary decoding

### Requirement: Round-trip

The reader SHALL reconstruct an equivalent `SKModelState` from the JSON representation.

#### Scenario: Round-trip fidelity

- **WHEN** a state is written and read back
- **THEN** metadata, tensor names, dtypes, shapes and values are identical

#### Scenario: Pretty and compact agree

- **WHEN** the same state is written in pretty and compact modes
- **THEN** both parse to the same state

### Requirement: Large tensors

The format SHALL document its memory behavior for large tensors and SHALL reject a tensor
whose values do not match its shape.

#### Scenario: Shape mismatch is rejected

- **WHEN** a tensor's flat value count differs from the shape product
- **THEN** the reader returns `ShapeMismatch`

### Requirement: Validation and concurrency

The reader SHALL reject unknown major versions, duplicate tensor names, invalid dtypes and
malformed JSON; concurrent reads SHALL agree.

#### Scenario: Malformed JSON

- **WHEN** the input is not valid JSON
- **THEN** the reader returns a structured conversion error

#### Scenario: Concurrent reads agree

- **WHEN** several threads read the same file
- **THEN** every state equals the single-threaded read
