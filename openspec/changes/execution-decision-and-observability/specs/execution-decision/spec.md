## Purpose

The automatic execution decision mechanism: a pure, deterministic resolver that turns an
execution intent (defaulting to `Automatic`) and a runtime context into a concrete execution
plan, and selects the active math backend for heavy dense algebra.

## ADDED Requirements

### Requirement: Automatic intent resolves to a concrete plan
The library SHALL resolve an `SKExecutionMode::Automatic` intent together with an
`SKExecutionContext` into a concrete `SKExecutionPlan` deterministically, so the caller can
run the operation without choosing a mode.

#### Scenario: Automatic intent yields a plan for small in-memory data
- **WHEN** an automatic intent is resolved against a context whose dataset fits in memory
- **THEN** the plan selects the in-process, synchronous mode with the available parallelism

#### Scenario: Automatic intent streams oversized sequential data
- **WHEN** an automatic intent is resolved against an oversized sequential-access dataset
- **THEN** the plan selects an out-of-core streaming mode with the declared batch size

### Requirement: Explicit incompatible intent is rejected
The library SHALL reject an explicit execution mode that is incompatible with the
algorithm's declared access pattern with a specific error, while automatic intent never
fails for that reason.

#### Scenario: Streaming is refused for a random-access algorithm
- **WHEN** an explicit out-of-core-streaming intent is applied to a random-access algorithm
- **THEN** resolution returns an incompatibility error naming the mode and pattern

### Requirement: Heavy dense algebra routes to the active backend
The library SHALL route heavy dense algebra to the selected `SKMathBackend`, choosing the
pure-Rust `faer` backend by default and the `ndarray-linalg` backend when the
`blas-backend` feature is enabled.

#### Scenario: Default build routes to the pure-Rust backend
- **WHEN** dense algebra is requested on a default build
- **THEN** it runs on the pure-Rust `faer` backend

#### Scenario: BLAS feature routes to the system backend
- **WHEN** dense algebra is requested on a build with the `blas-backend` feature enabled
- **THEN** it runs on the `ndarray-linalg` backend, with the pure-Rust path retained as a
  fallback

### Requirement: Resolution is pure and deterministic
The library SHALL make execution resolution a pure function of its inputs, so identical
intent and context always yield an identical plan, and the resolver is safe to run under
concurrency.

#### Scenario: Same inputs produce the same plan
- **WHEN** the same intent and context are resolved repeatedly
- **THEN** every resolution returns an identical plan

#### Scenario: Concurrent resolutions are independent
- **WHEN** the resolver is invoked concurrently from multiple threads
- **THEN** each call returns the plan for its own inputs with no shared mutable state