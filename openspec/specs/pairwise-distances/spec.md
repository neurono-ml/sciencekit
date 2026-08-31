## Purpose

Zero-copy pairwise-distance kernels (Euclidean, Manhattan, Cosine) used by neighbors,
SVM, clustering and imputation, computed with the SIMD-friendly squared-distance
formulation `||a−b||² = ||a||² − 2 a·b + ||b||²`.

## Requirements

### Requirement: Squared Euclidean distance via norms
The library SHALL compute the squared Euclidean distance matrix between two feature sets
using the norm-expansion formulation, avoiding a per-pair subtract loop.

#### Scenario: Distance matrix is symmetric and non-negative
- **WHEN** the squared distance between a set and itself is computed
- **THEN** the result is symmetric with a zero diagonal and all entries non-negative

#### Scenario: Manhattan distance is the L1 sum of absolute differences
- **WHEN** the Manhattan distance between two vectors is computed
- **THEN** it equals the sum of absolute per-coordinate differences

### Requirement: Cosine distance from normalized dot product
The library SHALL expose a cosine distance defined as `1 − cos(θ)` over row vectors.

#### Scenario: Identical vectors have zero cosine distance
- **WHEN** the cosine distance between two identical vectors is computed
- **THEN** it is zero (within tolerance)

#### Scenario: Orthogonal vectors have unit cosine distance
- **WHEN** the cosine distance between two orthogonal unit vectors is computed
- **THEN** it is one (within tolerance)