# Proposal

## Why

Sparse support now reaches every estimator — `SKFeatureBlock` carries CSR batches and the
scalers, imputers and linear models fit sparse streams without densifying — but the
pairwise-distance kernels still accept only dense `ArrayView2` inputs. Any consumer that
works on sparse features (neighbours, clustering, sparse SVM, evaluation helpers) must
densify first, which defeats the library's zero-copy and out-of-core promise exactly where
the data is largest. With the `metrics-foundation` change completing the dense catalogue
(Chebyshev and NaN-aware Euclidean), the remaining gap is a sparse path for the classic
metrics.

## What Changes

- Add sparse kernels for squared Euclidean, Euclidean, Manhattan, Chebyshev and Cosine over
  borrowed `sprs` CSR views, covering sparse×sparse, sparse×dense and dense×sparse inputs
  without ever densifying.
- Add a single dispatch surface, `SKPairwiseMetric` plus
  `sk_pairwise_distance_matrix(metric, left: SKDataView, right: SKDataView)`, which
  dispatches once per operation to the dense or sparse kernel and always returns a dense
  `Array2<F>`.
- Keep the existing dense functions unchanged as zero-copy fast paths; the sparse kernels
  are added beside them and the dispatcher composes both.
- Keep NaN-aware Euclidean dense-only: in CSR an implicit zero is an observed value, so
  there is no missing marker to compare. A sparse input for it returns
  `SKError::UnsupportedRepresentation` with a conversion suggestion.
- Validate shapes and representations in the dispatcher, returning `ShapeMismatch` instead
  of panicking; the existing dense entry points keep their current contract.
- Companion tests prove sparse equals dense on the same logical matrix, cover empty rows,
  zero norms, sparsity patterns, large inputs and concurrent use.

No estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `pairwise-distances`: adds sparse input support, the dense/sparse dispatch surface and
  the per-metric sparse semantics, and documents NaN-aware Euclidean as dense-only.

## Impact

- **Code:** `crates/sciencekit_math/src/pairwise` (new sparse modules and a dispatcher);
  no changes to existing dense function signatures.
- **APIs:** additive. `SKPairwiseMetric` and `sk_pairwise_distance_matrix` are new;
  `sk_sparse_*_distance_matrix` functions join the dense ones.
- **Dependencies:** none new; `sprs` is already a `sciencekit_math` dependency.
- **Downstream:** unblocks sparse neighbours and clustering, and lets sparse pipelines
  evaluate distances without a dense intermediate. `SKKNNImputer` remains on the dense
  NaN-aware path by semantics.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large sparse inputs, concurrent
  row-parallel execution, sparse/dense equivalence within the documented tolerance,
  companion `*_tests.rs` modules. Model export does not apply to a kernel.
