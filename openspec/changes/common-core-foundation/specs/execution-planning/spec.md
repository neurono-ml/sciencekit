## Purpose

Defines the PRD execution-mode decision mechanism (§5.3): separation between the user-declared intent and the plan resolved per operation, pure deterministic resolution with injectable context, and hard failure for explicit requests incompatible with the algorithm's access pattern.

## ADDED Requirements

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
