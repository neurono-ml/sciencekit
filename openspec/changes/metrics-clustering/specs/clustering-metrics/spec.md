# Spec Delta

## Purpose

Defines clustering evaluation: silhouette samples/score, adjusted Rand index, normalized
mutual information, homogeneity/completeness/V-measure, Davies-Bouldin and
Calinski-Harabasz, plus the silhouette scorer.

## ADDED Requirements

### Requirement: Silhouette

The library SHALL compute a silhouette value per sample and the mean silhouette score from
features and typed cluster assignments, excluding the explicit noise variant.

#### Scenario: Two well-separated clusters score near one

- **WHEN** two compact, well-separated clusters are evaluated
- **THEN** the mean silhouette is near `1`

#### Scenario: Noise is excluded

- **WHEN** the assignments contain `Noise`
- **THEN** those rows are excluded from the per-sample values and the mean

#### Scenario: A single cluster is rejected

- **WHEN** the assignments contain fewer than two clusters
- **THEN** the score returns `InvalidHyperparameter`

### Requirement: Adjusted Rand index

The library SHALL compute the adjusted Rand index between two labelings, correcting for
chance.

#### Scenario: Identical labelings

- **WHEN** two labelings are identical up to label permutation
- **THEN** the adjusted Rand index is `1`

#### Scenario: Random labelings

- **WHEN** two independent random labelings are compared
- **THEN** the index is near `0`

### Requirement: Normalized mutual information

The library SHALL compute normalized mutual information with arithmetic, geometric and max
normalizations, plus homogeneity, completeness and V-measure.

#### Scenario: Identical labelings

- **WHEN** two labelings are identical up to permutation
- **THEN** NMI, homogeneity, completeness and V-measure are `1`

#### Scenario: Normalization variants

- **WHEN** the same labelings are scored with each normalization
- **THEN** each result matches its documented formula

### Requirement: Davies-Bouldin and Calinski-Harabasz

The library SHALL compute the Davies-Bouldin index (lower is better) and the
Calinski-Harabasz index (higher is better) from features and assignments.

#### Scenario: Compact separated clusters

- **WHEN** two compact, well-separated clusters are scored
- **THEN** Davies-Bouldin is near zero and Calinski-Harabasz is large

#### Scenario: Noise is excluded

- **WHEN** the assignments contain `Noise`
- **THEN** those rows are excluded from both scores

### Requirement: Silhouette scorer

`SKSilhouetteScorer` SHALL implement `SKUnsupervisedScorer`, consuming typed assignments,
filtering noise and delegating to the free function.

#### Scenario: Scorer equals free function

- **WHEN** the scorer and the free function evaluate the same model output
- **THEN** both return the same value within tolerance

### Requirement: Representation parity

Silhouette, Davies-Bouldin and Calinski-Harabasz SHALL accept dense and sparse features
through the pairwise-distance/centroid path, dispatching once and never densifying.

#### Scenario: Sparse equals dense

- **WHEN** the same logical data is scored densely and as CSR
- **THEN** the scores agree within the documented tolerance

### Requirement: Validation and concurrency

The metrics SHALL reject empty input, length mismatch, fewer than two clusters, noise-only
input and non-finite values, and SHALL be deterministic under concurrent calls.

#### Scenario: Length mismatch is rejected

- **WHEN** assignments and features have different row counts
- **THEN** the metric returns `ShapeMismatch`

#### Scenario: Concurrent calls agree

- **WHEN** several threads compute the same score
- **THEN** every result equals the single-threaded result
