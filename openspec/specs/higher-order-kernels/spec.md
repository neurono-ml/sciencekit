## Purpose

Reusable higher-order operation kernels built on `azip!`/`par_azip!` and axis reductions,
honoring PRD §4.2 (higher-order functions mandatory, no manual index loops) and layout
awareness.

## Requirements

### Requirement: Elementwise and axis kernels via higher-order functions
The library SHALL provide kernels for elementwise transform, elementwise binary combine
and axis-wise reduction, implemented with `azip!`/`par_azip!`/`zip_mut_with`, never manual
index loops.

#### Scenario: Elementwise transform produces expected values
- **WHEN** an elementwise `x → 2x + 1` transform is applied to an array
- **THEN** every output element equals `2 * input + 1`

#### Scenario: Axis reduction is correct and layout-aware
- **WHEN** a column (axis 1) sum is computed on a row-major array
- **THEN** the result equals the per-column sum, computed efficiently over contiguous memory

### Requirement: In-place transformation avoids allocation
The library SHALL expose in-place transforms that mutate the input without allocating a
new buffer where the operation allows it.

#### Scenario: In-place scaling mutates without a copy
- **WHEN** an in-place scalar-multiply is applied
- **THEN** the input buffer is modified in place and no new array is returned

### Requirement: Cross-product reductions

`sciencekit_math` SHALL provide `sk_gram_matrix` and `sk_gram_vector`: given a 2-D array
`X` (rows × features) and, for the vector form, a 1-D target `y`, they return the cross
products `XᵀX` (a symmetric features × features matrix) and `Xᵀy` (a features vector),
staying generic over `F: SKFloat` and sharing the reduction grain.

#### Scenario: Gram matrix of a hand-checkable block

- **WHEN** `sk_gram_matrix` runs on a matrix with parallelism one
- **THEN** each output element is the dot product of the corresponding two columns

#### Scenario: Gram vector of a hand-checkable block

- **WHEN** `sk_gram_vector` runs on a matrix and a target vector
- **THEN** each output element is the dot product of the corresponding column with the
  target

#### Scenario: Both supported floats

- **WHEN** the reductions run on `f32` and on `f64` input
- **THEN** each returns its result in its own scalar type

### Requirement: The cross-product reductions honor the parallelism contract

`sk_gram_matrix` and `sk_gram_vector` SHALL execute their sequential form when
`parallelism` equals one, and otherwise dispatch across the compute pool through per-chunk
partial accumulators combined once at the end, with no shared mutable accumulator.

#### Scenario: Parallel matches sequential

- **WHEN** a large matrix is reduced with parallelism greater than one
- **THEN** the result equals the sequential reference within floating-point tolerance

#### Scenario: Gram matrix is symmetric

- **WHEN** `sk_gram_matrix` runs on any matrix
- **THEN** the result equals its own transpose

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

### Requirement: Kernels honor the resolved parallelism
Every dense kernel SHALL accept the resolved plan's `parallelism`: when it equals one the kernel SHALL execute its sequential form with no parallel dispatch, and when greater than one the kernel SHALL dispatch work across the compute pool; parallel results SHALL match the sequential reference within floating-point tolerance.

#### Scenario: Sequential path when parallelism is one
- **WHEN** a kernel is invoked with parallelism one on a large array
- **THEN** the result is correct and no parallel dispatch is incurred

#### Scenario: Parallel path matches sequential reference
- **WHEN** each kernel is invoked with parallelism greater than one on a large array
- **THEN** the result matches the sequential reference within floating-point tolerance

#### Scenario: Execution mode governs both directions
- **WHEN** the same kernel is driven through one resolved plan with parallelism one and another with parallelism greater than one, on identical input
- **THEN** the sequential and parallel outputs agree within floating-point tolerance, with no source-level duplication of the kernel

### Requirement: Race-free parallel axis reduction
The axis-0 column reduction SHALL parallelize over rows through per-thread partial accumulation combined into the output, without concurrent writes to a shared accumulator.

#### Scenario: Parallel column sum equals sequential column sum
- **WHEN** a wide, tall matrix is reduced along axis 0 with parallelism greater than one
- **THEN** the result equals the sequential column sums within floating-point tolerance

### Requirement: Documented grain with provenance
Each kernel's grain — the minimum elements per thread used to cap parallel dispatch — SHALL be recorded beside the kernel with its measurement provenance (benchmark protocol, machine, date), and the same grain SHALL apply unchanged to in-memory and streaming regimes.

#### Scenario: Grain provenance is consultable
- **WHEN** a reviewer inspects a kernel's grain constant
- **THEN** the accompanying provenance identifies how, on which machine and when the value was measured

#### Scenario: Re-calibration changes the number, not the contract
- **WHEN** the calibration benchmark is re-run on a different machine and the grain is updated
- **THEN** only the constant and its provenance change; kernel behavior contracts above remain unaffected

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
