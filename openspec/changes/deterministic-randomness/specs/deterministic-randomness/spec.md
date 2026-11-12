# Spec Delta

## Purpose

Defines the library's seeded randomness contract: the shared generator, how stochastic
estimators expose and derive seeds, and the reproducibility guarantees consumers can rely
on.

## ADDED Requirements

### Requirement: Shared seeded generator

The library SHALL expose a deterministic pseudo-random generator seeded by a `u64`, with
operations for raw integers, uniform floats, normal floats, shuffling and sampling without
replacement.

#### Scenario: Same seed, same stream

- **WHEN** two generators are created with the same seed
- **THEN** every sequence of draws is identical

#### Scenario: Uniform values are in range

- **WHEN** a uniform draw between `low` and `high` is requested
- **THEN** the result lies in `[low, high)`

#### Scenario: Shuffle is a permutation

- **WHEN** a slice is shuffled
- **THEN** it contains exactly the original elements and the order is reproducible for the
  same seed

### Requirement: Sub-seed derivation

The library SHALL derive independent component seeds from a global seed, a domain tag and a
component index, so trees, initializations, stages and folds do not share a stream.

#### Scenario: Component seeds differ

- **WHEN** two components with different indices derive from the same global seed
- **THEN** their derived seeds differ and each stream is reproducible

#### Scenario: Derivation is order-independent

- **WHEN** components are processed in a different order or on different threads
- **THEN** each component receives the same derived seed

### Requirement: Stochastic estimators expose a seed

Every estimator or algorithm with a stochastic fit SHALL expose `random_seed(u64)` on its
builder, defaulting to `0`, and SHALL derive every random choice from it.

#### Scenario: Default seed

- **WHEN** a stochastic estimator is built without a seed setter
- **THEN** it uses `0`

#### Scenario: Explicit seed

- **WHEN** a non-default seed is configured
- **THEN** every random choice in fit derives from that seed

#### Scenario: No ambient randomness

- **WHEN** a stochastic fit runs
- **THEN** it never reads a thread-local, time-based or address-based RNG

### Requirement: Deterministic collection and thread-count invariance

Results SHALL be collected in a stable order, and a fixed seed, input and configuration
SHALL produce the same result regardless of the number of worker threads.

#### Scenario: Thread count does not change the result

- **WHEN** the same fit runs with different rayon thread counts
- **THEN** the resulting model and predictions are identical, or within the documented
  floating-point tolerance where a parallel reduction order differs

#### Scenario: Stable order

- **WHEN** independent components finish in a different order
- **THEN** they are stored by index, not by completion order

### Requirement: Reproducibility scope

Reproducibility SHALL be guaranteed for the same library version, seed, input and backend;
cross-platform or cross-backend floating-point differences SHALL be documented and tested
with a tolerance rather than promised bit-for-bit.

#### Scenario: Documented scope

- **WHEN** a user reads the generator documentation
- **THEN** it states the same-seed guarantee and the cross-platform tolerance
