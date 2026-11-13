# halving-grid-search Specification

## Purpose
Defines `SKHalvingGridSearchCV`: successive-halving resource scheduling, candidate
evaluation and elimination, refit and validation.

## Requirements

### Requirement: Builder-only construction with documented defaults

The classifier and regressor halving searches SHALL be constructible only through their
builders, exposing a candidate list, splitter, scoring policy, `factor` (default 3),
`minimum_resource` (default 1), `maximum_resource` (default all samples), `resource`
(`NumberOfSamples` default), `aggressive_elimination` (default false), `random_seed`
(default 0), `refit` (default true), `error_score` and `execution_mode`.

#### Scenario: Default configuration

- **WHEN** a halving search is built without optional setters
- **THEN** it uses factor 3, sample resources, no aggressive elimination, seed 0 and refit

### Requirement: Successive halving loop

Fit SHALL evaluate the surviving candidates on the current resource, keep the best
`ceil(n / factor)`, increase the resource by `factor` and repeat until one candidate or the
maximum resource is reached.

#### Scenario: Candidate count shrinks

- **WHEN** nine candidates are searched with factor 3
- **THEN** the surviving counts follow `9 → 3 → 1`

#### Scenario: Resource grows

- **WHEN** the minimum resource is 10 and the factor is 3
- **THEN** the resource schedule is `10, 30, 90, ...` up to the maximum

### Requirement: Deterministic elimination

Candidates SHALL be ranked by mean score and ties SHALL be broken by candidate index, so
the elimination trace is reproducible.

#### Scenario: Tie is deterministic

- **WHEN** two candidates share a mean score at a round
- **THEN** the smaller candidate index survives

#### Scenario: Seed controls subsampling

- **WHEN** the same seed and data are searched twice
- **THEN** every resource subsample and elimination trace is identical

### Requirement: Resource subsampling

When the resource is smaller than the data, the search SHALL draw a deterministic subset
of the training rows for each round and keep class proportions for classification when the
splitter requires it.

#### Scenario: Subset size follows the resource

- **WHEN** the current resource is 30 rows
- **THEN** each candidate is evaluated on exactly 30 rows per fold

#### Scenario: Classification stratification is preserved

- **WHEN** a stratified splitter is used with a subsample
- **THEN** the subsample preserves the class proportions

### Requirement: Aggressive elimination

When `aggressive_elimination` is enabled, rounds whose minimum resource cannot support the
remaining candidates SHALL be skipped by pruning clearly inferior candidates early,
according to the documented rule.

#### Scenario: Aggressive mode skips a round

- **WHEN** the resource cannot support all surviving candidates and aggressive elimination
  is enabled
- **THEN** the search prunes candidates before the next round

### Requirement: Refit and prediction

When `refit` is true, the winner SHALL be refitted on the full data and the wrapper SHALL
delegate prediction to it; when false, prediction returns `UnsupportedOperation`.

#### Scenario: Best model is available

- **WHEN** the search completes with refit
- **THEN** the wrapper predicts through the refitted winner

#### Scenario: No refit blocks prediction

- **WHEN** `refit` is false
- **THEN** prediction returns `UnsupportedOperation`

### Requirement: Validation and concurrency

The search SHALL reject `factor < 2`, invalid resource bounds, empty candidates, an
incompatible scorer and invalid groups, and SHALL be deterministic under concurrency.

#### Scenario: Invalid factor is rejected

- **WHEN** `factor` is less than 2
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Resource bounds are validated

- **WHEN** `minimum_resource` exceeds `maximum_resource`
- **THEN** construction returns `InvalidHyperparameter`
