# Design

## Context

See `proposal.md` — Why. `sciencekit_math::pairwise` owns the dense kernels
(Euclidean/Manhattan/Cosine and, after `metrics-foundation`, Chebyshev and NaN-aware
Euclidean). `sciencekit_math` already depends on `sciencekit_common`, so the shared
`SKDataView` enum (non-exhaustive `Dense`/`Sparse`) is available as the dispatch seam.
`SKDataView::as_dense` documents the project rule: a dense-only consumer rejects a sparse
input once, before processing any element. The sparse streaming work gives consumers CSR
data through `SKFeatureBlock`, so the missing piece is a sparse pairwise path.

## Goals / Non-Goals

**Goals:**
- Add sparse kernels for squared Euclidean, Euclidean, Manhattan, Chebyshev and Cosine.
- One dispatch surface over `SKDataView`, dispatching once per operation.
- Keep every input borrowed and every output dense; never densify a sparse input.
- Keep the existing dense functions as unchanged zero-copy fast paths.
- Structured `ShapeMismatch`/`UnsupportedRepresentation` errors at the dispatch surface.

**Non-Goals:**
- A sparse NaN-aware Euclidean path; CSR implicit zeros are observed values.
- Sparse output matrices (pairwise distances are generally dense).
- Changing the existing dense function signatures or the estimator APIs.
- A generic open metric trait or dynamic dispatch.

## Decisions

### Decision 1 — Additive API: dense fast paths, sparse kernels, one dispatcher

The existing `sk_*_distance_matrix` dense functions stay unchanged. Beside them, the
crate gains `sk_sparse_squared_euclidean_distance_matrix`,
`sk_sparse_euclidean_distance_matrix`, `sk_sparse_manhattan_distance_matrix`,
`sk_sparse_chebyshev_distance_matrix` and `sk_sparse_cosine_distance_matrix`, all taking
borrowed `sprs::CsMatView`. On top sits:

```rust
pub enum SKPairwiseMetric {
    SquaredEuclidean,
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine,
    NanEuclidean,
}

pub fn sk_pairwise_distance_matrix<'a, F: SKFloat>(
    metric: SKPairwiseMetric,
    left: SKDataView<'a, F>,
    right: SKDataView<'a, F>,
) -> Result<Array2<F>, SKError>;
```

The dispatcher matches the metric, then matches the two representations (four
combinations), and calls the specialized kernel. The enum is closed, like `SKBackendKind`,
so the match is exhaustive and there is no dynamic dispatch.

Alternatives considered: a generic `SKPairwiseKernel` trait (open and object-safe, but
unnecessary for a closed catalogue and harder to keep exhaustive); overloading one
function through `Into` (Rust has no overloading, and it would hide which representation
was selected).

### Decision 2 — Dispatch once, never per element

The representation match happens once at the dispatch surface; the chosen kernel then
reads its representation directly. No inner loop branches on `Dense`/`Sparse`. This follows
the `data-view-boundary` rule and keeps the hot loops monomorphic.

### Decision 3 — Sparse algorithms

Rows are canonical CSR rows with ascending stored indices; the sparse kernels merge two
sorted index lists, so the work is proportional to the stored entries, not to the feature
count (except where noted for sparse×dense Chebyshev).

- **Squared Euclidean / Euclidean** — norm expansion
  `‖a‖² + ‖b‖² − 2·Σ_{shared} aᵢbᵢ`. Norms sum the stored squares; the dot product is a
  sorted merge over shared indices. Sparse×dense computes the same dot over the sparse
  indices and precomputes the dense row norms once. Euclidean is the square root.
- **Manhattan** — merge the sorted indices: `|aᵢ−bᵢ|` on shared coordinates and `|aᵢ|` or
  `|bᵢ|` on one-sided coordinates. Sparse×dense precomputes the dense row's L1 norm once
  and adjusts it at the sparse indices, so each pair costs `O(nnz_sparse)`.
- **Chebyshev** — the same merge, but with a maximum: shared coordinates contribute
  `|aᵢ−bᵢ|`, one-sided coordinates contribute the absolute stored value. Sparse×dense
  scans the dense row (`O(features)`) because a dense maximum may sit at an index the
  sparse row does not store; this is documented as the one non-`O(nnz)` path.
- **Cosine** — `1 − dot/(‖a‖·‖b‖)` from the stored dot and norms; a zero norm keeps the
  dense policy (distance one).
- **NaN-aware Euclidean** — not implemented for sparse. The dispatcher returns
  `UnsupportedRepresentation` when either side is sparse, naming the missing-marker reason
  and suggesting densification.

Alternatives considered: densify then call the dense kernel (defeats the point and breaks
the memory promise); a single generic row-iterator abstraction shared with the dense
kernels (the representations are too different, and the formulas are short enough to keep
two tested paths).

### Decision 4 — Errors at the dispatch surface, fast paths unchanged

The dispatcher validates the feature counts before computing and returns `ShapeMismatch`
with both dimensions. It also returns `UnsupportedRepresentation` for a sparse
NaN-aware request. The specialized dense and sparse functions keep the existing
assert-based contract, so direct callers pay no `Result` plumbing and the dense API does
not break. Empty inputs are valid: a zero-row side yields a matrix with that zero
dimension.

### Decision 5 — Zero-copy inputs, dense output

Both inputs are borrowed views (`ArrayView2` or `CsMatView`), never converted into owned
arrays. The output is always a dense row-major `Array2<F>`: distances between arbitrary
pairs are generally non-zero, so a sparse output would not pay for its structure.

### Decision 6 — Row-parallel work with the dense grain

The sparse kernels parallelize over the left rows with rayon, using the same grain as the
dense kernels (a pair count below the grain stays sequential). Each row writes to its own
output row, so there are no locks and the result is deterministic. The public functions are
pure and thread-safe.

### Decision 7 — Module layout

```
crates/sciencekit_math/src/pairwise/
  dispatcher.rs              SKPairwiseMetric + sk_pairwise_distance_matrix
  sparse/
    mod.rs                   pure dispatcher
    row_merge.rs             sorted-row merge helpers
    squared_euclidean.rs
    manhattan.rs
    chebyshev.rs
    cosine.rs
    *_tests.rs
  chebyshev.rs, nan_euclidean.rs, ...   existing dense kernels unchanged
```

`mod.rs` files stay pure dispatchers; every companion `*_tests.rs` is declared at the end
of its dispatcher.

## Risks / Trade-offs

- **Norm-expansion cancellation for near-identical sparse rows** → the dense kernels
  already live with this; tests use the documented tolerance and include near-identical
  rows. A direct accumulation fallback can be added later if a real workload demands it.
- **Sparse×dense Chebyshev is `O(features)` per pair** → documented; the merge path is
  still allocation-free, and sparse×sparse stays `O(nnz)`.
- **CSR index order** → the kernels assume canonical ascending indices. `sprs` CSR rows
  are canonical by construction; a caller building a non-canonical view must normalize it,
  and the boundary documents this.
- **Two implementations of the same formula** → the sparse/dense equivalence tests are the
  contract; every metric has a same-logical-matrix test.
- **API surface growth** → the dispatcher is the single public entry for generic
  consumers; the representation-specific functions are the fast paths.

## Migration Plan

Additive only: new functions and an enum, no existing signature changes. Consumers opt in
by calling the dispatcher or the sparse functions; removing the change means removing the
new modules.

## Open Questions

None blocking. A GPU/OpenCL pairwise kernel and a sparse NaN-aware convention (explicit NaN
storage or a missing mask) are separate future changes.
