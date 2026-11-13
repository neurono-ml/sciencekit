# Proposal

## Why

The PRD records sparse out-of-core streaming as delivered through the owned
`SKFeatureBlock` contract, but delivery of the contract does not guarantee every estimator's
`fit_streaming`/`partial_fit` actually accepts sparse batches. A coverage audit and the
missing paths are needed so users can stream CSR data through every algorithm that claims
streaming support.

## What Changes

- Audit every estimator with `fit_streaming`/`partial_fit` and record dense/sparse coverage
  in a matrix.
- Add sparse batch handling to the algorithms that lack it, using `SKFeatureBlock::Sparse`
  and existing sparse kernels without densifying.
- Ensure streaming batch order, multi-epoch reopenable providers and early stop behave
  identically for dense and sparse batches.
- Add per-family sparse streaming tests: single batch, multiple batches, partial fit,
  large sparse input, concurrency and equivalence with the dense path.
- Document memory bounds and any algorithm that intentionally remains dense-only (with a
  structured error).
- No new algorithms, no `memmap2` and no GPU.

## Capabilities

### New Capabilities

- `sparse-streaming-coverage`: the dense/sparse coverage contract for every streaming
  estimator and its acceptance matrix.

### Modified Capabilities

None.

## Impact

- **Code:** streaming modules across `sciencekit_preprocessing`, `sciencekit_impute`,
  `sciencekit_linear_model` and any future streaming estimator.
- **APIs:** additive sparse support; dense behavior unchanged.
- **Dependencies:** none new.
- **Downstream:** makes the sparse out-of-core promise real and testable.
- **Acceptance criteria:** every streaming estimator accepts CSR batches or documents a
  deliberate dense-only exception; no densification; memory bounded.
