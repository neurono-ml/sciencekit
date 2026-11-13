# Spec Delta

## MODIFIED Requirements

### Requirement: Precise rejection of unsupported representation
Algorithms supporting only part of the representations SHALL reject the others at runtime with a specific unsupported-representation error, guiding the consumer toward the proper conversion; representation dispatch SHALL happen exactly once per operation, never per element. Rejection applies both to an algorithm that declares dense-only support under the dense-only exception and to one that rejects a representation whose mathematics is semantically undefined (for example, a NaN-aware distance over implicit CSR zeros). The rejection requirement does not by itself admit a dense-only algorithm: admission is governed by "Dense-only exception with a public alternative".

#### Scenario: Sparse rejected by a dense-only algorithm
- **WHEN** sparse data is delivered to an algorithm declaring dense-only support
- **THEN** the operation fails before processing any element, with an error indicating the mismatch and the suggested conversion path

#### Scenario: Semantically undefined representation is rejected
- **WHEN** a representation has no defined meaning for the operation (for example a NaN-aware distance over implicit CSR zeros)
- **THEN** the operation rejects it up front with the unsupported-representation error and no alternative path is required

## ADDED Requirements

### Requirement: Representation parity for tolerant algorithms

An algorithm that mathematically tolerates more than one feature representation SHALL
accept dense, sparse and mixed dense/sparse inputs on every public operation, dispatch on
the representation exactly once per operation and never densify a sparse input.

#### Scenario: Dense and sparse inputs produce the same result

- **WHEN** the same logical feature matrix is passed once densely and once as CSR
- **THEN** the operation succeeds on both and produces the same result within the
  documented tolerance

#### Scenario: Mixed inputs are accepted

- **WHEN** a two-matrix operation receives a dense left view and a sparse right view, or
  the reverse
- **THEN** the operation reads each representation directly and succeeds

#### Scenario: Sparse input is not densified

- **WHEN** a wide sparse matrix with few stored entries is passed to a tolerant algorithm
- **THEN** the operation completes without materializing a dense copy of that matrix

### Requirement: Dense-only exception with a public alternative

An algorithm whose mathematics genuinely requires dense geometry MAY declare dense-only
support only when its public estimator surface still covers dense, sparse and mixed
inputs through a documented alternative path, and it SHALL reject the unsupported
representation with `UnsupportedRepresentation` and a conversion suggestion.

#### Scenario: Dense-only kernel rejects sparse input

- **WHEN** a sparse view reaches a dense-only kernel
- **THEN** the operation returns `UnsupportedRepresentation` before processing any element

#### Scenario: Estimator still serves sparse input

- **WHEN** the estimator built on that dense-only kernel is fitted or predicts on sparse
  data
- **THEN** it dispatches to the documented alternative path and succeeds without
  densifying the input

#### Scenario: The exception is documented

- **WHEN** a change invokes the dense-only exception
- **THEN** its proposal states why the mathematics requires dense geometry and names the
  alternative path that keeps estimator-level parity

### Requirement: Mixed representation for two-matrix operations

Every two-matrix operation that supports representation parity SHALL handle all four
combinations (dense×dense, sparse×sparse, sparse×dense, dense×sparse) with the same
observable result, validating shape and representation before computing.

#### Scenario: Every representation combination agrees

- **WHEN** the same logical data is supplied as dense×dense, sparse×sparse, sparse×dense
  and dense×sparse
- **THEN** all four results agree within the documented tolerance

#### Scenario: Shape mismatch is rejected before computation

- **WHEN** the two inputs declare incompatible shapes
- **THEN** the operation returns `ShapeMismatch` and computes no element
