# feature-union Specification

## Purpose
Defines `SKFeatureUnion`: fitting several transformers on the same input, concatenating
their outputs and exposing the union as a pipeline stage.

## Requirements

### Requirement: Union composition

`SKFeatureUnion` SHALL compose two or more transformer stages that receive the same input
and SHALL expose the union as a transformer stage usable inside `SKPipeline`.

#### Scenario: Two members concatenate

- **WHEN** a scaler and a one-hot encoder are unioned
- **THEN** the output contains the scaler columns followed by the encoder columns

#### Scenario: Empty union is rejected

- **WHEN** a union is built with no members
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Fit and transform

Fit SHALL fit every member on the same input block; transform SHALL transform the input
with every fitted member and concatenate the results in member order.

#### Scenario: Every member sees the input

- **WHEN** a union is fitted
- **THEN** every member is fitted on the same input rows and columns

#### Scenario: Transform is deterministic

- **WHEN** the same fitted union transforms the same input twice
- **THEN** the output is identical

### Requirement: Representation policy

All-dense member outputs SHALL concatenate to a dense block; any sparse member output SHALL
make the union output CSR. No member SHALL densify a sparse input it declares to accept.

#### Scenario: All-dense output

- **WHEN** every member outputs dense
- **THEN** the union output is dense

#### Scenario: Sparse member forces CSR

- **WHEN** one member outputs sparse
- **THEN** the union output is CSR and dense members are inserted as stored columns

### Requirement: Member errors and shapes

A member error SHALL be surfaced with the member index; row counts SHALL match across
members; the output feature count SHALL equal the sum of member widths.

#### Scenario: Row count mismatch

- **WHEN** two members return different row counts
- **THEN** the union returns `ShapeMismatch`

#### Scenario: Member failure

- **WHEN** a member fails
- **THEN** the union returns that error and does not produce a partial output

### Requirement: Accessors and concurrency

The fitted union SHALL expose the member count, per-member widths and the total feature
count, and SHALL be safely shareable.

#### Scenario: Accessors describe the union

- **WHEN** a union is fitted
- **THEN** the widths sum to the total feature count

#### Scenario: Concurrent transform agrees

- **WHEN** several threads transform with the same fitted union
- **THEN** every result equals the sequential result
