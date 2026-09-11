## Purpose

Generic prefetch pipeline over a sequential streaming source (`SKLazySource`): owns the I/O ∥ CPU overlap required by PRD §4.4/§5.1 — reading the next owned batch while the compute pool processes the current one — so algorithms provide only a per-batch update step.

## ADDED Requirements

### Requirement: Prefetch overlap between input and computation
When streaming, the driver SHALL issue the read of the next batch concurrently with the computation of the current batch, bounded by a buffer of at most the plan-resolved depth, so that input and computation overlap rather than alternate.

#### Scenario: Reader advances while compute runs
- **WHEN** the update step of batch k takes longer than the source takes to produce batch k+1
- **THEN** the source has already been asked for batch k+1 before batch k's computation completes (observed via the source's read log in test)

#### Scenario: Buffer depth is respected
- **WHEN** the plan resolves a buffer depth of one (double buffering)
- **THEN** at most two batches are resident at any moment (the one computing and the one prefetched)

### Requirement: Ordered delivery of owned batches
Batches SHALL be delivered for computation in source order, each fully owned and usable after the source has moved on, preserving the `streaming-batches` contract.

#### Scenario: Batch survives the handoff
- **WHEN** a batch is handed to the update step and the source advances
- **THEN** the batch's data remains intact and correctly ordered with respect to the sequence position recorded at extraction

### Requirement: Per-batch update closure with early stop
The driver SHALL delegate every batch to an algorithm-supplied update step that receives the batch and the algorithm state, and SHALL stop reading and computing when that step signals completion.

#### Scenario: Early stop halts the pipeline
- **WHEN** the update step returns stop after batch k (for example on convergence)
- **THEN** batch k's effect is kept, no batch beyond k is requested from the source, and the call returns without error

### Requirement: Structured error propagation from the source
A read failure SHALL surface the central taxonomy error to the caller, keeping the effects of already-processed batches, and the driver SHALL not swallow, retry nor panic.

#### Scenario: Intermediate read failure stops with the taxonomy error
- **WHEN** reading an intermediate batch fails
- **THEN** the driver returns the structured error, updates already applied to earlier batches remain in the algorithm state, and no further batches are processed

### Requirement: Final batch delivered exactly once
On a finite source consumed to the end, the driver SHALL deliver exactly one final batch, identified as final, and then terminate normally.

#### Scenario: Finite stream ends cleanly
- **WHEN** the last batch of a finite source is produced
- **THEN** the update step receives it flagged as final and the driver returns success afterwards without reading further

### Requirement: Plan-driven compute parallelism on the compute side
The compute side of the driver SHALL apply the resolved plan's parallelism to each batch's computation, dispatching work to the compute pool dedicated to CPU processing and never blocking it with input waits.

#### Scenario: Parallelism comes from the resolved plan
- **WHEN** the plan resolves compute parallelism greater than one for a streaming operation
- **THEN** each batch's computation uses that parallelism level, and the returned results match the sequential execution within floating-point tolerance
