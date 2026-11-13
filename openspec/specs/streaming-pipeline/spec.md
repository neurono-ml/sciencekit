# streaming-pipeline Specification

## Purpose
Defines the streaming pipeline: streaming transformer fitting, per-batch transformation,
epochs, the finalizer policy and error propagation.

## Requirements

### Requirement: Streaming transformer contract

A streaming transformer SHALL fit from a sequence of owned batches and SHALL transform one
owned batch at a time without retaining the whole dataset.

#### Scenario: Batch transform

- **WHEN** a fitted streaming transformer receives a batch
- **THEN** it returns a transformed batch with the same row count

#### Scenario: Fit consumes a source

- **WHEN** a streaming transformer is fitted
- **THEN** it consumes the source through the shared driver and stores only its learned
  state

### Requirement: Chained fit

`SKStreamingPipeline` SHALL fit stages in order, each on the transformed output of the
previous stage, and SHALL support multiple epochs when the provider is reopenable.

#### Scenario: Two-stage fit

- **WHEN** two streaming transformers are fitted
- **THEN** the second stage sees the first stage's transformed batches

#### Scenario: One-shot provider

- **WHEN** a non-reopenable provider supplies a multi-stage pipeline
- **THEN** fitting either materializes within the configured memory budget or returns a
  structured error naming the reopenable requirement

#### Scenario: Multi-epoch

- **WHEN** a reopenable provider is configured for more than one epoch
- **THEN** the source is reopened per epoch and the epoch count is recorded

### Requirement: Per-batch prediction/transform

After fitting, the pipeline SHALL transform a batch through every fitted stage in order,
preserving dense/sparse representation and never densifying a sparse batch.

#### Scenario: Streaming equals batch transform

- **WHEN** the same data is transformed through the streaming pipeline and through the
  equivalent linear batch pipeline
- **THEN** the outputs agree within the documented tolerance

#### Scenario: Sparse batch flow

- **WHEN** a sparse batch passes through sparse-capable stages
- **THEN** the block stays CSR

### Requirement: Finalizer policy

The pipeline SHALL support a batch estimator finalizer fitted on the accumulated
transformed blocks within a documented memory budget; a streaming estimator finalizer is a
future extension and SHALL be reported as unsupported.

#### Scenario: Batch finalizer

- **WHEN** the final stage is a batch estimator and the accumulated output fits the budget
- **THEN** the estimator is fitted on the accumulated blocks

#### Scenario: Memory budget exceeded

- **WHEN** the accumulated output exceeds the configured budget
- **THEN** fitting returns a structured error instead of exhausting memory

### Requirement: Error propagation

A source read error or a stage error SHALL stop the pipeline and surface the structured
error; already-applied state is not rolled back.

#### Scenario: Source failure

- **WHEN** the source fails mid-stream
- **THEN** the pipeline returns the central error and stops reading

#### Scenario: Stage failure

- **WHEN** a stage rejects a batch
- **THEN** the pipeline returns that error with the stage index

### Requirement: Validation and concurrency

The pipeline SHALL reject an empty stage list, incompatible stages, invalid epochs and an
exhausted one-shot provider on a multi-epoch request, and SHALL be safely shareable after
fitting.

#### Scenario: Empty pipeline is rejected

- **WHEN** a streaming pipeline is built without stages
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Concurrent transform agrees

- **WHEN** several threads transform with the same fitted pipeline
- **THEN** every result equals the sequential result
