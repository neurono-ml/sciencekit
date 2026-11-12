# Spec Delta

## Purpose

Defines the dense/sparse coverage contract for streaming estimators and the acceptance
matrix that proves sparse batches are processed without densification.

## ADDED Requirements

### Requirement: Sparse streaming batches

Every estimator that exposes `fit_streaming` or `partial_fit` over feature batches SHALL
accept both dense and CSR blocks through `SKFeatureBlock`.

#### Scenario: CSR batch accepted

- **WHEN** a CSR block is delivered to a streaming estimator
- **THEN** the operation succeeds and produces the same model as the dense equivalent
  within tolerance

#### Scenario: Dense behavior unchanged

- **WHEN** the same estimator receives dense blocks
- **THEN** its results are unchanged

### Requirement: No densification

The sparse path SHALL not materialize a dense copy of the block; the only dense state
allowed is the algorithm's own small parameter state.

#### Scenario: Wide sparse input

- **WHEN** a wide CSR block with few stored entries is streamed
- **THEN** no dense copy of the block is allocated

### Requirement: Streaming semantics

Batch order, multi-epoch reopenable providers, early stop and error propagation SHALL be
identical for dense and sparse blocks.

#### Scenario: Multi-epoch sparse stream

- **WHEN** a reopenable provider supplies CSR batches for multiple epochs
- **THEN** epochs are processed in order and the model equals the dense run

#### Scenario: Read error

- **WHEN** the source fails mid-stream
- **THEN** the structured error is returned and prior batches remain applied

### Requirement: Coverage matrix

The library SHALL document, per estimator, whether streaming is dense-only, sparse-only or
both, and a dense-only estimator SHALL reject CSR with `UnsupportedRepresentation`.

#### Scenario: Dense-only exception

- **WHEN** a documented dense-only streaming estimator receives CSR
- **THEN** it returns `UnsupportedRepresentation` with guidance

#### Scenario: Coverage is tested

- **WHEN** a new streaming estimator is added
- **THEN** the coverage matrix and tests require an explicit dense/sparse decision

### Requirement: Memory and concurrency

Sparse streaming SHALL keep memory bounded by the batch and algorithm state, and concurrent
streams SHALL not interfere.

#### Scenario: Bounded memory

- **WHEN** a large sparse stream is processed
- **THEN** memory stays proportional to the batch plus state, not the dataset

#### Scenario: Concurrent streams

- **WHEN** several streams run concurrently
- **THEN** each produces its own model without shared mutable state

The completeness guarantee is enforced by a source scan that walks the sibling crates at
test time and counts the `fit_streaming` and `partial_fit` definitions, not by comparing the
matrix with another hand-written list.
