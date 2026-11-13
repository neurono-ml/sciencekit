# Spec Delta

## Purpose

Defines a deterministic randomized range finder and truncated singular-value decomposition
for dense and sparse matrices, used by PCA and TruncatedSVD.

## ADDED Requirements

### Requirement: Randomized range finder

The library SHALL compute a low-rank approximation of a matrix by projecting it onto a
random subspace, orthonormalizing the projected basis and optionally applying power
iterations before the final decomposition.

#### Scenario: Rank-k approximation reconstructs a low-rank matrix

- **WHEN** a matrix of rank `r` is factorized with `k >= r`
- **THEN** the reconstruction error is within the documented tolerance

#### Scenario: Power iterations improve a decaying spectrum

- **WHEN** the same matrix is factorized with more power iterations
- **THEN** the reconstruction error does not increase

### Requirement: Truncated singular-value result

The library SHALL return the left singular vectors, the non-increasing singular values and
the right singular vectors of the requested rank, with shapes consistent with the input.

#### Scenario: Result shapes

- **WHEN** an `m × n` matrix is factorized to `k` components
- **THEN** `U` is `m × k`, the singular-value vector has length `k` and `V` is `n × k`

#### Scenario: Singular values are non-negative and sorted

- **WHEN** any matrix is factorized
- **THEN** every singular value is non-negative and the sequence is non-increasing

### Requirement: Dense and sparse input parity

The factorization SHALL accept dense and sparse feature views, dispatch once per operation
and never densify a sparse input.

#### Scenario: Sparse equals dense

- **WHEN** the same logical matrix is factorized densely and as CSR
- **THEN** the singular values and reconstructed matrices agree within the documented
  tolerance

#### Scenario: Sparse input is not densified

- **WHEN** a wide sparse matrix with few stored entries is factorized
- **THEN** the computation succeeds without materializing a dense copy of the input

### Requirement: Optional implicit centering

The factorization SHALL accept an optional per-feature centering vector. When supplied, it
SHALL compute `Xc = X − 1μᵀ` implicitly through the projected products and SHALL NOT
materialize a dense centered matrix, for dense or sparse input.

#### Scenario: Centered dense equals explicit centering

- **WHEN** a dense matrix is factorized with a centering vector
- **THEN** the result equals factorizing the explicitly centered matrix within the
  documented tolerance

#### Scenario: Centered sparse avoids densification

- **WHEN** a sparse matrix with a non-zero centering vector is factorized
- **THEN** the computation succeeds without materializing the dense centered matrix

#### Scenario: Centered sparse equals centered dense

- **WHEN** the same logical matrix is factorized with the same centering vector densely and
  as CSR
- **THEN** the singular values and reconstructed centered matrices agree within the
  documented tolerance

### Requirement: Deterministic projection and signs

The random projection SHALL be derived from the caller's seed, and the singular-vector
signs SHALL follow a fixed convention, so repeated calls return identical results.

#### Scenario: Repeated calls agree

- **WHEN** the same input and seed are factorized twice
- **THEN** `U`, the singular values and `V` are identical

#### Scenario: Sign convention is fixed

- **WHEN** a singular vector has a largest-magnitude entry
- **THEN** that entry is positive; a tie is resolved by the smallest row index

### Requirement: Validation

The library SHALL reject a zero or excessive component count, zero power iterations, an
invalid oversampling parameter and empty input with structured errors before computing.

#### Scenario: Excessive components are rejected

- **WHEN** `k` is larger than `min(m, n)`
- **THEN** the call returns `InvalidHyperparameter`

#### Scenario: Empty input is rejected

- **WHEN** the matrix has zero rows or zero columns
- **THEN** the call returns `EmptyInput`

#### Scenario: Invalid oversampling is rejected

- **WHEN** the oversampling parameter is zero or negative
- **THEN** the call returns `InvalidHyperparameter`

### Requirement: Concurrent factorization

The factorization SHALL be pure and thread-safe, and concurrent calls with the same input
and seed SHALL return identical results.

#### Scenario: Concurrent calls agree

- **WHEN** several threads factorize the same matrix with the same seed
- **THEN** every result equals the single-threaded result
