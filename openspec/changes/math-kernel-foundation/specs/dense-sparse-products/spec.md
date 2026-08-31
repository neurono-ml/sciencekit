## Purpose

Sparse (`sprs`) product kernels (sparse×dense, sparse×sparse) with zero-copy views,
essential for Lasso, linear SVMs and text classification (PRD §4.5).

## ADDED Requirements

### Requirement: Sparse×dense product via views
The library SHALL compute a CSR sparse matrix times a dense vector/matrix using zero-copy
`CsMatView` inputs, without densifying the sparse operand.

#### Scenario: Sparse×dense product equals dense reference
- **WHEN** a CSR matrix is multiplied by a dense vector
- **THEN** the result equals the product computed by densifying and multiplying

#### Scenario: Sparse operand is not densified
- **WHEN** the product is computed
- **THEN** the sparse matrix remains a view over its original storage (no copy made)

### Requirement: Sparse×sparse product
The library SHALL support sparse×sparse products through the `sprs` product API.

#### Scenario: Sparse×sparse matches structural expectation
- **WHEN** two sparse matrices are multiplied
- **THEN** the non-zero structure and values match the reference product