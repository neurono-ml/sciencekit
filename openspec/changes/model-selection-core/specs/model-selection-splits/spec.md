# Spec Delta

## Purpose

Defines reproducible data partitioning: train/test splitting, K-fold, stratified K-fold,
group K-fold, the index-split contract, row gathering and cross-validation scoring.

## ADDED Requirements

### Requirement: Train/test split

The library SHALL split features and optional targets into train and test partitions by a
fraction or an explicit count, with optional shuffling and a seeded order.

#### Scenario: Fractional split

- **WHEN** a 20% test split is requested on 100 rows
- **THEN** the test partition has 20 rows and the train partition 80

#### Scenario: Explicit count

- **WHEN** a test count of 15 is requested
- **THEN** the test partition has 15 rows

#### Scenario: Shuffle is reproducible

- **WHEN** the same seed and input are split twice
- **THEN** the train and test index sets are identical

### Requirement: Stratified split

When a target view is supplied for stratification, the split SHALL preserve each class's
proportion in both partitions.

#### Scenario: Class proportions are preserved

- **WHEN** a binary target is split 50/50 with a 20% test fraction
- **THEN** each partition contains both classes in approximately the original proportion

#### Scenario: Continuous targets are rejected for stratification

- **WHEN** stratification is requested with continuous targets
- **THEN** the split returns `UnsupportedRepresentation`

### Requirement: K-fold

`SKKFold` SHALL produce `number_of_splits` index splits whose test partitions are disjoint
and cover every row exactly once.

#### Scenario: Every row is tested once

- **WHEN** a 5-fold split covers 50 rows
- **THEN** each row appears in exactly one test partition and four train partitions

#### Scenario: Shuffling is seeded

- **WHEN** shuffle is enabled with a seed
- **THEN** repeated splits produce identical partitions

### Requirement: Stratified K-fold

`SKStratifiedKFold` SHALL preserve class proportions within each fold.

#### Scenario: Each fold sees every class

- **WHEN** a multiclass target is split into stratified folds
- **THEN** every fold's test partition contains every class in proportion to the data

#### Scenario: Too few members per class is rejected

- **WHEN** a class has fewer members than the split count
- **THEN** the splitter returns `InvalidHyperparameter`

### Requirement: Group K-fold

`SKGroupKFold` SHALL keep every group entirely in one fold and never split a group across
train and test.

#### Scenario: Groups do not leak

- **WHEN** groups are split into folds
- **THEN** no group appears in both the train and test side of a fold

#### Scenario: Split count defaults to the group count

- **WHEN** no split count is configured
- **THEN** the number of distinct groups is used

### Requirement: Index split contract

Every split SHALL be represented as train and test index vectors, and the splitter
contract SHALL accept optional targets and groups so one interface serves unsupervised,
stratified and group splitters.

#### Scenario: Indices address the original rows

- **WHEN** a split is applied to a feature matrix
- **THEN** every index lies in the original row range and the two vectors are disjoint

#### Scenario: Optional inputs are ignored when unused

- **WHEN** a plain K-fold splitter receives targets or groups
- **THEN** the split is unchanged

### Requirement: Row gathering

The library SHALL gather selected rows from dense and sparse feature blocks and targets
into owned partitions without densifying a sparse input.

#### Scenario: Dense gathering

- **WHEN** arbitrary row indices are gathered from a dense matrix
- **THEN** the result contains the selected rows in index order

#### Scenario: Sparse gathering

- **WHEN** arbitrary row indices are gathered from a CSR matrix
- **THEN** the result is CSR with the selected rows and no dense copy

### Requirement: Cross-validation scoring

The library SHALL provide a cross-validation score helper for classifier and regressor
pipelines, fitting one model per fold and returning the per-fold scores.

#### Scenario: One score per fold

- **WHEN** a 5-fold cross-validation runs
- **THEN** five scores are returned in fold order

#### Scenario: Fold parallelism is deterministic

- **WHEN** the same cross-validation runs with different thread counts
- **THEN** the scores are identical within the documented tolerance

### Requirement: Validation

The splitters SHALL reject empty input, invalid fractions, too many splits, class/group
counts smaller than the split count, length mismatches and non-finite values.

#### Scenario: Too many splits

- **WHEN** the split count exceeds the row count
- **THEN** the splitter returns `InvalidHyperparameter`

#### Scenario: Group length mismatch

- **WHEN** the group vector length differs from the row count
- **THEN** the splitter returns `ShapeMismatch`
