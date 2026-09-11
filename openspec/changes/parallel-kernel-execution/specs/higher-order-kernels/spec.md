## ADDED Requirements

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
