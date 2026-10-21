## ADDED Requirements

### Requirement: Size-aware parallelism resolution
Resolution SHALL derive the plan's `parallelism` from the mode, the size of the work unit, the available cores and the documented per-kernel grain — not unconditionally from the core count: the work unit is the whole dataset for in-memory modes, one batch for streaming, and an arbitrary shard for memory-mapped access; `parallelism` SHALL be capped at `unit_size / grain` (rounded up), capped by the core count, and SHALL be exactly one when the machine is single-core or the unit is below one grain.

#### Scenario: Small dataset resolves sequential
- **WHEN** an in-memory operation resolves over a context whose dataset size yields fewer than one grain of elements per thread
- **THEN** the plan's parallelism is exactly one, so the operation runs sequentially without parallel-dispatch overhead

#### Scenario: Streaming resolution uses the batch as the unit
- **WHEN** a streaming operation resolves with a batch hint whose size yields fewer than one grain of elements per thread
- **THEN** the plan's parallelism is one even though the whole dataset is larger than memory, because each batch is too small to amortize dispatch overhead

#### Scenario: Memory-mapped access resolves full sharding
- **WHEN** a memory-mapped random-access operation resolves
- **THEN** the plan's parallelism equals the core count, since shards can execute independently

#### Scenario: Deterministic resolution per operation
- **WHEN** resolution runs twice in one operation with identical intent and context
- **THEN** both invocations yield the same parallelism, regardless of machine load between them

### Requirement: Double-buffer memory guard for streaming
A streaming resolution SHALL verify that the double-buffered pipeline fits in memory — twice the batch size against available memory — and SHALL fail with a structured error naming both quantities when the batch hint violates it; the error is raised before any data processing.

#### Scenario: Oversized batch hint refused before processing
- **WHEN** a streaming operation resolves with a batch hint whose doubled size exceeds simulated available memory
- **THEN** resolution fails with that structured error and no batch has been read

#### Scenario: Automatic intent never trips the guard alone
- **WHEN** intent is automatic and the resolved batch fits the memory guard
- **THEN** resolution succeeds and produces the plan without manual reconfiguration
