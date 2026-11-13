# data-view-boundary Specification

## Purpose
Defines the library's data boundary: the canonical representations of features (dense/sparse) and targets (continuous/integer/nominal), the universal conversion mechanism accepted at public inputs, the zero-copy guarantee of native conversions, non-breaking evolution through new variants, and the label canonicalization supporting classifiers and codecs.

## Requirements

### Requirement: Dense/sparse feature view as an extensible enum
The library SHALL expose a single feature view type covering dense and sparse representations, marked as non-exhaustive to allow new variants without breaking consumers using wildcard matching.

#### Scenario: Borrowed dense matrix enters without copying
- **WHEN** an existing dense matrix is passed to an operation as a borrowed view
- **THEN** no copy of the elements is performed by the conversion

#### Scenario: Sparse matrix enters through the same boundary
- **WHEN** a sparse matrix in compressed-row format is passed to an operation
- **THEN** the conversion produces the sparse variant of the view referencing the same data

#### Scenario: Owned intermediate block enters by borrowing
- **WHEN** the owned result of a previous pipeline stage is passed to the next stage
- **THEN** the conversion borrows the block without duplicating its data

### Requirement: Universal conversion seam at public inputs
Every public data input SHALL be declared over the standard library's fallible conversion trait, so that: types with infallible conversion are accepted automatically via the standard library's blanket implementation, and external integrators can plug in their own types by implementing conversions — including fallible ones — into the library's view types.

#### Scenario: User type with Into works without friction
- **WHEN** a consumer passes to the API a type implementing only infallible conversion into the view
- **THEN** the call compiles and runs without additional code, because the std automatically promotes that conversion to the fallible form required by the bound

#### Scenario: Third-party fallible conversion reports a structured error
- **WHEN** an external integrator implements a fallible conversion from its type into the view and it fails during a call
- **THEN** the error reaches the consumer through the Result of the operation itself, in the conversion-failure variant, without panicking

### Requirement: Precise rejection of unsupported representation
Algorithms supporting only part of the representations SHALL reject the others at runtime with a specific unsupported-representation error, guiding the consumer toward the proper conversion; representation dispatch SHALL happen exactly once per operation, never per element. Admission under the dense-only exception is governed by "Dense-only exception with a public alternative".

#### Scenario: Sparse rejected by a dense-only algorithm
- **WHEN** sparse data is delivered to an algorithm declaring dense-only support
- **THEN** the operation fails before processing any element, with an error indicating the mismatch and the suggested conversion path

#### Scenario: Semantically undefined representation is rejected
- **WHEN** a representation has no defined meaning for the operation (for example a NaN-aware distance over implicit CSR zeros)
- **THEN** the operation rejects it up front with the unsupported-representation error and no alternative path is required

### Requirement: Continuous/integer/nominal target view
The library SHALL expose a single target view type with three representations — continuous values, integers and nominal (textual symbols) — marked non-exhaustive; integers SHALL be elevatable to continuous losslessly when the continuous interpretation is legitimate.

#### Scenario: Integer targets elevated for regression
- **WHEN** targets stored as integers are provided to a continuous-reading context
- **THEN** elevation to floating point happens losslessly and without manual intervention

#### Scenario: Nominal targets reference borrowed text
- **WHEN** textual labels are provided as targets
- **THEN** the nominal view references the original strings without copying them

### Requirement: Deterministic label canonicalization
The library SHALL offer canonicalization of nominal/integer/boolean targets into compact indices accompanied by a reversible table, deterministic for the same input sequence, serving as the foundation for classifier automatic encoding and future explicit codecs.

#### Scenario: Roundtrip preserves original labels
- **WHEN** a sequence of labels is canonicalized and the resulting indices are decoded through the produced table
- **THEN** the original label sequence is fully restored

#### Scenario: Same input produces the same table
- **WHEN** the same label sequence is canonicalized twice
- **THEN** the produced label→index mappings are identical to each other

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
