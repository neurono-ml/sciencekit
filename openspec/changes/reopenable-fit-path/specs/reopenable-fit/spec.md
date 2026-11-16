# reopenable-fit Specification

## Purpose

Lets iterative estimators train on datasets larger than memory by re-reading the same
logical data across passes, without changing in-memory behavior.

## ADDED Requirements

### Requirement: Iterative estimators stream over reopenable sources

Every estimator whose declared access pattern is `Iterative` SHALL expose `fit_streaming`
accepting a reopenable source, and SHALL reject a source that cannot be reopened before the
first pass completes.

#### Scenario: KMeans converges over a reopened stream

- **WHEN** `SKKMeans` is fitted with `fit_streaming` over a reopenable provider
- **THEN** each Lloyd pass observes the same logical rows in the same order
- **AND** the fitted centers equal the in-memory fit within the documented tolerance

#### Scenario: Non-reopenable source is rejected before the first pass

- **WHEN** an `Iterative` estimator receives a source that yields each row once
- **THEN** the fit fails with a structured error naming the reopening requirement and the
  source has not been drained for a second pass

### Requirement: Deterministic epochs and early stop

Each pass SHALL see identical data, the epoch count SHALL be observable, and early stop
SHALL be respected identically for dense and CSR batches.

#### Scenario: Early stop halts an iterative fit

- **WHEN** a convergence condition holds mid-epoch
- **THEN** the fit stops with the epoch count recorded and no further batch is requested

#### Scenario: Dense and CSR passes agree

- **WHEN** the same logical data streams as dense batches and as CSR batches
- **THEN** both fits produce the same model within the documented tolerance

### Requirement: Parity and unchanged in-memory path

The reopenable fit SHALL equal the in-memory fit of the same logical data within the
documented tolerance, and the in-memory fit SHALL keep its behavior, tests and signatures.

#### Scenario: In-memory callers unchanged

- **WHEN** an existing caller uses `fit` with an in-memory view
- **THEN** nothing changes in signature or result

#### Scenario: Oversized dataset trains without memory violation

- **WHEN** the dataset exceeds the plan's memory guard for in-memory modes
- **THEN** the automatic resolution selects streaming and the fit completes within the
  double-buffer guard
