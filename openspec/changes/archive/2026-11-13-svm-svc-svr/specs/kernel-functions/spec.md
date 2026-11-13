# Spec Delta

## Purpose

Defines the kernel functions used by support-vector machines and later kernel methods:
the mathematical catalogue, the dense/sparse/mixed representation contract, the dense
output and the validation rules shared by every kernel.

## ADDED Requirements

### Requirement: Kernel catalogue

The library SHALL expose linear, polynomial, radial-basis-function, sigmoid and
precomputed kernels over pairs of feature matrices.

#### Scenario: Linear kernel is the inner product

- **WHEN** the linear kernel is computed for two rows
- **THEN** the result equals their dot product

#### Scenario: Polynomial kernel applies degree and offset

- **WHEN** the polynomial kernel is computed with `gamma`, `coef0` and `degree`
- **THEN** the result equals `(gamma · <x, y> + coef0)^degree`

#### Scenario: Radial-basis-function kernel uses squared distance

- **WHEN** the RBF kernel is computed with `gamma`
- **THEN** the result equals `exp(-gamma · ||x − y||²)`

#### Scenario: Sigmoid kernel applies the hyperbolic tangent

- **WHEN** the sigmoid kernel is computed with `gamma` and `coef0`
- **THEN** the result equals `tanh(gamma · <x, y> + coef0)`

#### Scenario: Precomputed kernel accepts a square matrix

- **WHEN** a precomputed kernel matrix is supplied
- **THEN** the library uses the supplied values and validates that the matrix is square

### Requirement: Representation parity for kernel inputs

Every kernel SHALL accept dense, sparse and mixed dense/sparse inputs, dispatch once per
operation and never densify a sparse input.

#### Scenario: Sparse and dense inputs agree

- **WHEN** the same logical matrix is represented once densely and once as CSR
- **THEN** both representations produce the same kernel matrix within the documented
  tolerance

#### Scenario: Mixed representation is supported

- **WHEN** one side is dense and the other is sparse
- **THEN** the kernel reads each representation directly and returns the kernel matrix

#### Scenario: Sparse input is not densified

- **WHEN** a wide sparse matrix with few stored entries is passed to a kernel
- **THEN** the computation succeeds without materializing a dense copy

### Requirement: Dense kernel output

Every kernel SHALL return a dense row-major matrix with one row per left input row and
one column per right input row.

#### Scenario: Output shape follows the inputs

- **WHEN** a kernel is computed for `m` left rows and `n` right rows
- **THEN** the returned matrix has shape `(m, n)`

#### Scenario: Inputs stay borrowed

- **WHEN** a caller passes borrowed dense or CSR views
- **THEN** the call consumes no owned copy of either input buffer

### Requirement: Kernel parameter validation

The library SHALL reject a negative `gamma`, a non-positive polynomial degree, and a
precomputed matrix that is not square, with structured errors before computing values.

#### Scenario: Negative gamma is rejected

- **WHEN** a kernel is configured with `gamma < 0`
- **THEN** the call returns `InvalidHyperparameter` naming `gamma`

#### Scenario: Non-square precomputed matrix is rejected

- **WHEN** a precomputed kernel receives a non-square matrix
- **THEN** the call returns `ShapeMismatch`

#### Scenario: Feature-count mismatch is rejected

- **WHEN** the two kernel inputs declare different feature counts
- **THEN** the call returns `ShapeMismatch` with both dimensions

### Requirement: Kernel numerical stability

The library SHALL compute the RBF kernel through the squared-distance form and SHALL
clamp tiny negative squared distances caused by floating-point cancellation to zero.

#### Scenario: Identical rows have unit RBF similarity

- **WHEN** the RBF kernel is computed for a row and itself
- **THEN** the result is `1` within the documented tolerance

#### Scenario: Distant rows have negligible RBF similarity

- **WHEN** two rows are far apart relative to `gamma`
- **THEN** the result is `0` within the documented tolerance and never NaN

### Requirement: Parallel kernel execution

The library SHALL evaluate independent kernel rows in parallel with rayon and SHALL
produce the same result as the sequential path.

#### Scenario: Parallel equals sequential

- **WHEN** a large kernel matrix is computed sequentially and through the row-parallel
  path
- **THEN** both matrices agree within the documented tolerance

#### Scenario: Concurrent kernel calls are deterministic

- **WHEN** several threads compute the same kernel matrix simultaneously
- **THEN** every result equals the single-threaded result
