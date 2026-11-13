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

### Requirement: Chebyshev distance

The library SHALL compute the Chebyshev distance matrix between two dense feature sets,
where each entry is the maximum absolute coordinate difference of the corresponding pair.

#### Scenario: Identical rows have zero distance

- **WHEN** the Chebyshev distance between a set and itself is computed
- **THEN** the diagonal is zero and the matrix is symmetric

#### Scenario: The largest coordinate dominates

- **WHEN** two rows differ by `1` in one coordinate and by `3` in another
- **THEN** their Chebyshev distance is `3`

#### Scenario: Distance is non-negative

- **WHEN** any two rows are compared
- **THEN** the resulting distance is greater than or equal to zero

### Requirement: NaN-aware Euclidean distance

The library SHALL compute a Euclidean distance over the features observed in both rows,
scaling the sum by the ratio of the total feature count to the co-observed feature count:
`sqrt(Σ_present (aᵢ − bᵢ)² · total_features / present_features)`. A pair with no
co-observed feature SHALL yield NaN.

#### Scenario: Missing coordinates are excluded and compensated

- **WHEN** two rows agree on every co-observed coordinate while other coordinates are missing
- **THEN** their distance is zero

#### Scenario: The scale compensates for fewer co-observed features

- **WHEN** two rows differ on the only co-observed feature out of four total features
- **THEN** the distance equals the absolute difference multiplied by two

#### Scenario: No co-observed feature yields NaN

- **WHEN** every coordinate of at least one row is missing
- **THEN** the distance for that pair is NaN

#### Scenario: Present rows behave like Euclidean distance

- **WHEN** both rows have every coordinate present
- **THEN** the distance equals the standard Euclidean distance

#### Scenario: The matrix is symmetric

- **WHEN** the NaN-aware distance is computed between two feature sets
- **THEN** the result is symmetric
