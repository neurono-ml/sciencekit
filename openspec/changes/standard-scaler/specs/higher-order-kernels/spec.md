## ADDED Requirements

### Requirement: Sum-of-squares axis reduction

`sciencekit_math` SHALL provide `sk_axis_sum_squares`, the `Σx²` companion of
`sk_axis_sum`: it reduces a 2-D array along axis 0 (per-column) or axis 1 (per-row) and
returns a 1-D result, sharing `SK_AXIS_SUM_GRAIN` and staying generic over `F: SKFloat`.

#### Scenario: Column totals

- **WHEN** a matrix is reduced along axis 0 with parallelism one
- **THEN** each output element is the sum of squares of that column

#### Scenario: Row totals

- **WHEN** a matrix is reduced along axis 1
- **THEN** each output element is the sum of squares of that row

#### Scenario: Both supported floats

- **WHEN** the reduction runs on `f32` and on `f64` input
- **THEN** each returns totals in its own scalar type

### Requirement: The reduction honors the parallelism contract

`sk_axis_sum_squares` SHALL execute its sequential form when `parallelism` equals one, and
otherwise dispatch across the compute pool through per-chunk partial accumulators combined
once at the end, with no shared mutable accumulator.

#### Scenario: Parallel matches sequential

- **WHEN** a wide, tall matrix is reduced along axis 0 with parallelism greater than one
- **THEN** the result equals the sequential reference within floating-point tolerance

#### Scenario: Per-row parallel matches sequential

- **WHEN** a non-square matrix is reduced along axis 1 with parallelism greater than one
- **THEN** the result has one element per row and equals the sequential reference

