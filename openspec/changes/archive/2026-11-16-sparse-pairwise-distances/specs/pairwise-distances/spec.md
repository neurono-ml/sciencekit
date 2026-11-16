# Spec Delta

## ADDED Requirements

### Requirement: Dense and sparse inputs share one dispatch surface

The library SHALL accept dense or sparse feature views on either side of a pairwise
distance computation, select the kernel once per operation, and return a dense distance
matrix without densifying either input.

#### Scenario: Sparse rows compare with sparse rows

- **WHEN** both sides of a pairwise computation are sparse CSR views
- **THEN** the computation uses the sparse kernel and produces the full dense distance matrix

#### Scenario: Sparse rows compare with dense rows

- **WHEN** one side is a sparse CSR view and the other is a dense view
- **THEN** the kernel reads each representation directly and produces the dense distance matrix

#### Scenario: Sparse input is never densified

- **WHEN** a sparse matrix with far fewer stored entries than coordinates is passed
- **THEN** the computation succeeds without materializing a dense copy of that matrix

### Requirement: Sparse squared Euclidean and Euclidean distances

The library SHALL compute sparse squared Euclidean distances from stored values only,
using the norm expansion and a dot product over shared stored coordinates, and Euclidean
distance as the square root of that form.

#### Scenario: Sparse equals dense on the same logical matrix

- **WHEN** the same logical matrix is represented once as dense and once as CSR
- **THEN** both representations produce the same squared Euclidean and Euclidean matrices within the documented tolerance

#### Scenario: Empty sparse rows have zero distance to each other

- **WHEN** two rows have no stored entries
- **THEN** their squared Euclidean and Euclidean distances are zero

#### Scenario: A sparse set is symmetric with zero diagonal

- **WHEN** a sparse set is compared with itself
- **THEN** the result is symmetric with a zero diagonal and non-negative entries

### Requirement: Sparse Manhattan distance

The library SHALL compute Manhattan distances over sparse rows by merging sorted stored
coordinates, charging implicit zeros as observed zeros.

#### Scenario: Sparse Manhattan equals dense

- **WHEN** the same logical matrix is represented as dense and as CSR
- **THEN** both representations produce the same Manhattan matrix within the documented tolerance

#### Scenario: One-sided stored coordinates contribute their absolute value

- **WHEN** a coordinate is stored in one row and implicit in the other
- **THEN** its contribution to the Manhattan distance is the absolute stored value

### Requirement: Sparse Chebyshev distance

The library SHALL compute Chebyshev distances over sparse rows by merging sorted stored
coordinates, with implicit zeros participating in the maximum.

#### Scenario: Sparse Chebyshev equals dense

- **WHEN** the same logical matrix is represented as dense and as CSR
- **THEN** both representations produce the same Chebyshev matrix within the documented tolerance

#### Scenario: A one-sided stored coordinate contributes its absolute value

- **WHEN** a coordinate is stored in one row and implicit in the other
- **THEN** its absolute value participates in the maximum

#### Scenario: Identical sparse rows have zero distance

- **WHEN** two rows have the same stored coordinates and values
- **THEN** their Chebyshev distance is zero

### Requirement: Sparse cosine distance

The library SHALL compute cosine distance over sparse rows from the stored dot product and
the stored norms, preserving the dense zero-norm policy.

#### Scenario: Sparse cosine equals dense

- **WHEN** the same logical matrix is represented as dense and as CSR
- **THEN** both representations produce the same cosine matrix within the documented tolerance

#### Scenario: A zero-norm sparse row keeps the dense policy

- **WHEN** one of the compared rows has no stored entries
- **THEN** the cosine distance is one, as in the dense kernel

#### Scenario: Sparse cosine is scale invariant

- **WHEN** a sparse row is multiplied by a positive scalar
- **THEN** its cosine distance to an unchanged row is unchanged

### Requirement: NaN-aware Euclidean is dense-only

The library SHALL reject a sparse input for the NaN-aware Euclidean distance with
`UnsupportedRepresentation`, because an implicit CSR zero is an observed value and no
missing marker exists in the sparse representation.

#### Scenario: Sparse input is rejected with guidance

- **WHEN** a sparse view is passed to a NaN-aware Euclidean computation
- **THEN** the operation returns `UnsupportedRepresentation` with a conversion suggestion and computes nothing

#### Scenario: Dense NaN-aware Euclidean keeps working

- **WHEN** dense views carrying NaN are passed to the NaN-aware kernel
- **THEN** the co-observed-feature distance is computed as before

### Requirement: Dispatch validates shapes before computing

The dispatch surface SHALL reject a feature-count mismatch with `ShapeMismatch` before
computing any distance, and SHALL never panic on a representation or shape mismatch.

#### Scenario: Mismatched feature counts are rejected

- **WHEN** the two inputs declare different feature counts
- **THEN** the operation returns `ShapeMismatch` with both dimensions and computes no distance

#### Scenario: Empty inputs produce an empty matrix

- **WHEN** either side has zero rows
- **THEN** the result has the corresponding zero dimension and no error is raised

### Requirement: Distance output is dense and inputs stay borrowed

Every pairwise API SHALL return a dense row-major distance matrix regardless of the input
representation, and SHALL borrow both inputs without copying their element buffers.

#### Scenario: Sparse inputs produce a dense matrix

- **WHEN** a sparse pairwise computation succeeds
- **THEN** the result is a dense matrix with one row per left row and one column per right row

#### Scenario: Inputs are borrowed

- **WHEN** a caller passes borrowed dense or CSR views
- **THEN** the call consumes no owned copy of either input

### Requirement: Sparse distance computation is row-parallel and deterministic

Sparse distance matrices SHALL be computed with row-parallel work sharing the dense
kernels' grain, and SHALL produce the same result as the sequential path.

#### Scenario: Parallel equals sequential

- **WHEN** the same large sparse input is computed sequentially and through the row-parallel path
- **THEN** both matrices are equal within the documented tolerance

#### Scenario: Concurrent calls are deterministic

- **WHEN** several threads compute the same sparse distance matrix simultaneously
- **THEN** every result equals the single-threaded result
