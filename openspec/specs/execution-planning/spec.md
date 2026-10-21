# execution-planning Specification

## Purpose
Defines the PRD execution-mode decision mechanism (§5.3): separation between the user-declared intent and the plan resolved per operation, pure deterministic resolution with injectable context, and hard failure for explicit requests incompatible with the algorithm's access pattern.

## Requirements

### Requirement: Intent and plan are distinct concepts
The library SHALL distinguish the consumer-declared execution intent — including automatic as default mode — from the effective plan resolved at operation time, which consolidates chosen mode, parallelism and batch size.

#### Scenario: Automatic intent produces a concrete plan
- **WHEN** an operation runs with automatic intent over a known context
- **THEN** a concrete plan is produced before any data processing

#### Scenario: Explicit intent preserved when compatible
- **WHEN** the consumer explicitly declares a mode compatible with the algorithm's access pattern
- **THEN** the resolved plan reflects exactly that mode

### Requirement: Pure, deterministic, injectable resolution
Resolution SHALL be a pure function over the intent and an explicit context — available memory, CPU cores, dataset size, access pattern declared by the algorithm and optional batch hint — with no global state access; the context SHALL be providable by the caller, enabling deterministic tests independent of the machine.

#### Scenario: Same context produces same plan
- **WHEN** resolution runs twice with identical intent and context
- **THEN** the resulting plans are identical in every field

#### Scenario: Simulated context dispenses with a real machine
- **WHEN** tests exercise resolution with simulated memory and core values
- **THEN** the obtained plans reflect exclusively the simulated values, without reading the physical environment

### Requirement: Hard failure for explicit incompatibility
Explicitly requesting a mode incompatible with the declared access pattern SHALL fail with a specific error naming both requested mode and declared pattern, checked before processing any data; automatic mode SHALL NEVER produce such an error.

#### Scenario: Sequential streaming refused for a random-access algorithm
- **WHEN** the consumer explicitly requests sequential streaming for an algorithm whose declared pattern is random access
- **THEN** the operation fails immediately with the incompatibility error, identifying both sides of the conflict

#### Scenario: Automatic never conflicts with the declared pattern
- **WHEN** intent is automatic, whatever the context and algorithm
- **THEN** resolution always picks a mode compatible with the declared pattern and never fails on incompatibility

### Requirement: Resolution happens per operation
Each heavy operation SHALL resolve its own plan from the stored intent and that moment's context — fit and prediction resolve independently, because data size only becomes known at each operation's input.

#### Scenario: Prediction on larger volume than fit resolves its own plan
- **WHEN** fit processes a small in-memory set and subsequent prediction receives volume exceeding simulated available memory
- **THEN** prediction's plan differs from fit's plan, reflecting the new context without manual reconfiguration

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
