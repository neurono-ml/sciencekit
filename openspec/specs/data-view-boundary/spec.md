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
Algorithms supporting only part of the representations SHALL reject the others at runtime with a specific unsupported-representation error, guiding the consumer toward the proper conversion; representation dispatch SHALL happen exactly once per operation, never per element.

#### Scenario: Sparse rejected by a dense-only algorithm
- **WHEN** sparse data is delivered to an algorithm declaring dense-only support
- **THEN** the operation fails before processing any element, with an error indicating the mismatch and the suggested conversion path

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

