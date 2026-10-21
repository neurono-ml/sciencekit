# Spec Delta

## ADDED Requirements

### Requirement: Min/max axis reduction

`sciencekit_math` SHALL provide a per-column extremum reduction returning both the minimum and
the maximum of each column in one pass, sharing the reduction grain and staying generic over
`F: SKFloat`. It SHALL honor the dense-kernel parallelism contract: `parallelism = 1` runs the
sequential form, `parallelism > 1` dispatches across the compute pool via per-chunk partial
extrema combined once at the end, with no shared mutable accumulator.

#### Scenario: Column extrema

- **WHEN** a matrix is reduced along its columns
- **THEN** each output pair holds that column's minimum and maximum

#### Scenario: Parallel matches sequential

- **WHEN** a wide, tall matrix is reduced with parallelism greater than one
- **THEN** the result equals the sequential reference exactly (min/max are order-independent)

#### Scenario: Both supported floats

- **WHEN** the reduction runs on `f32` and on `f64` input
- **THEN** each returns extrema in its own scalar type
