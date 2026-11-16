# Proposal

## Why

The execution plan already knows how to stream oversized data through
`OutOfCoreStreaming`, and the streaming driver already serves reopenable providers, but
the iterative estimators still refuse a streamed input: `SKKMeans`, `SKPCA`,
`SKTruncatedSVD`, `SKKNNImputer` and `SKMissingIndicator` have no `fit_streaming`, and
their in-memory-only `fit` means a dataset larger than RAM cannot be trained even though
the mathematics only needs several passes over the same rows. The conformance audit
records this as the iterative-side counterpart of the mappable gap.

## What Changes

- Add `fit_streaming` over reopenable sources (`SKReopenableSource` /
  `SKSupervisedReopenableSource` providers) for every estimator whose declared access
  pattern is `Iterative`: `SKKMeans`, `SKPCA` (full, randomized and covariance solvers),
  `SKTruncatedSVD`, `SKKNNImputer`, `SKMissingIndicator`.
- Guarantee every pass sees the same logical data: a non-reopenable source is rejected
  before the first pass with a structured error.
- Preserve the existing single-pass streaming estimators untouched; the change is additive.
- Guarantee parity within the documented tolerance between the in-memory fit and the
  reopenable streaming fit of the same logical data, on dense and CSR layouts.
- Explicit non-goals: mappable fit (`mappable-fit-path`), sparse-batch streaming
  (`sparse-streaming-batches`), GPU backends, new estimators, changing in-memory results.

## Capabilities

### New Capabilities

- `reopenable-fit`: the multi-pass streaming contract for iterative estimators, the
  reopening guarantee, the parity harness and the rejection of non-reopenable sources.

### Modified Capabilities

None.

## Impact

- `sciencekit_common`: reopenable validation helper and the multi-pass parity harness.
- `sciencekit_cluster`, `sciencekit_decomposition`, `sciencekit_impute`: the new
  `fit_streaming` paths.
- The DAG gains the change in the `contracts` lane after `mappable-fit-path`.
