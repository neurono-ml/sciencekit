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