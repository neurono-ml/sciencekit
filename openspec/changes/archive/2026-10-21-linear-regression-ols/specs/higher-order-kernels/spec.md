## ADDED Requirements

### Requirement: Cross-product reductions

`sciencekit_math` SHALL provide `sk_gram_matrix` and `sk_gram_vector`: given a 2-D array
`X` (rows × features) and, for the vector form, a 1-D target `y`, they return the cross
products `XᵀX` (a symmetric features × features matrix) and `Xᵀy` (a features vector),
staying generic over `F: SKFloat` and sharing the reduction grain.

#### Scenario: Gram matrix of a hand-checkable block

- **WHEN** `sk_gram_matrix` runs on a matrix with parallelism one
- **THEN** each output element is the dot product of the corresponding two columns

#### Scenario: Gram vector of a hand-checkable block

- **WHEN** `sk_gram_vector` runs on a matrix and a target vector
- **THEN** each output element is the dot product of the corresponding column with the
  target

#### Scenario: Both supported floats

- **WHEN** the reductions run on `f32` and on `f64` input
- **THEN** each returns its result in its own scalar type

### Requirement: The cross-product reductions honor the parallelism contract

`sk_gram_matrix` and `sk_gram_vector` SHALL execute their sequential form when
`parallelism` equals one, and otherwise dispatch across the compute pool through per-chunk
partial accumulators combined once at the end, with no shared mutable accumulator.

#### Scenario: Parallel matches sequential

- **WHEN** a large matrix is reduced with parallelism greater than one
- **THEN** the result equals the sequential reference within floating-point tolerance

#### Scenario: Gram matrix is symmetric

- **WHEN** `sk_gram_matrix` runs on any matrix
- **THEN** the result equals its own transpose
