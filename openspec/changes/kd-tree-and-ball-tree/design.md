# Design

## Context

See `proposal.md` — Why. `sciencekit_math` exposes borrowed dense pairwise-distance
kernels and the shared `SKDataView` boundary; `sciencekit_common` owns the builder state,
error taxonomy, execution mode and observability. There is no neighbours crate yet, no
`memmap2` integration and no persistence layer. `sparse-pairwise-distances` (in flight)
supplies the mixed-representation distance dispatcher that the brute-force searcher will
use in W3.3; this change only needs the dense kernels.

## Goals / Non-Goals

**Goals:**
- Two exact, immutable dense spatial indexes with one shared query contract.
- Deterministic results independent of thread count.
- Builder-only construction with the project-wide backend and execution-mode surface.
- Validation and observability consistent with the rest of the library.

**Non-Goals:**
- Sparse tree geometry; sparse and mixed inputs are served by the brute-force searcher in
  W3.3.
- Automatic algorithm selection between KD-tree, ball tree and brute force (W3.3).
- `memmap2` backing, persistence, ONNX/safetensors export (Wave 7).
- Approximate nearest-neighbor indexes.

## Decisions

### Decision 1 — New crate, trees are indexes not estimators

The indexes live in a new `sciencekit_neighbors` crate rather than in `sciencekit_math`
because they are a consumer of distances, not a numeric primitive, and because
`SKKNeighborsClassifier`/`Regressor` and `SKDBSCAN` will share them. They are not
estimators: they have no `fit`/`predict` contract, no target input and no fitted model
distinct from the index. They still use builder-only construction, `SKBackendKind` and
`SKExecutionMode` so the selection surface stays uniform.

Alternative considered: putting the trees in `sciencekit_math` (rejected — math would then
own algorithm state) and exposing free `sk_*` constructors (rejected — violates the
builder rule).

### Decision 2 — Array-based nodes over a point permutation

The tree owns an index permutation `Vec<usize>` and a flat `Vec<SKTreeNode<F>>`. A node
stores its contiguous `[start, end)` range in the permutation, optional left/right child
indices, and its geometry. Leaves point directly at their range, so a leaf scan is a
contiguous walk over indices without pointer chasing. This mirrors scikit-learn's
array-based trees and keeps the structure cache-friendly and trivially `Send + Sync`.

Alternative considered: `Box`-linked nodes (rejected — allocation per node, poor cache
locality, harder to snapshot).

### Decision 3 — KD-tree splits on the widest feature, ball tree on two farthest pivots

The KD-tree picks the feature with the largest spread in the node and partitions at the
median with `select_nth_unstable_by`, comparing `(value, training_index)` so equal values
never make the split non-deterministic. The ball tree picks two pivots (the point farthest
from the node centroid and the point farthest from it), partitions by the closer pivot,
and stores the centroid and covering radius. Both stop at `leaf_size` or zero spread.

Alternatives considered: variance-based feature selection (more expensive, no measurable
gain on the target workloads); random projections (approximate, out of scope).

### Decision 4 — Dense geometry, explicit sparse rejection

KD bounding boxes and ball centroids/radii are dense vectors of length `n_features`. A
sparse KD-tree would need sparse per-node bounds and a sparse centroid, which either costs
`O(nodes · features)` anyway or loses pruning quality. The index therefore rejects sparse
training and query views with `UnsupportedRepresentation` and points at the brute-force
searcher. Representation parity for the *estimators* is delivered in W3.3 by dispatching
dense work to trees and sparse/mixed work to brute force.

Alternative considered: densifying sparse input inside the tree (rejected — defeats the
zero-copy/out-of-core promise exactly where data is largest).

### Decision 5 — Closed metric catalogue with metric-specific bounds

`SKNeighborMetric` is a closed enum (`Euclidean`, `SquaredEuclidean`, `Manhattan`,
`Chebyshev`). Each variant supplies a point distance, a leaf scan and a node lower bound:
axis-aligned box distance for KD, `max(0, d(q, centroid) − radius)` for ball. Cosine and
NaN-aware Euclidean are rejected because tree pruning is only sound for metrics obeying
the triangle inequality and because the sparse NaN representation has no missing marker.
A closed enum keeps the hot loop monomorphic and the match exhaustive.

Alternative considered: a public `SKDistanceMetric` trait (open, but dynamic dispatch in
the inner loop and no compile-time exhaustiveness; the brute-force searcher can still use
the math dispatcher).

### Decision 6 — One shared searcher contract

`SKNearestNeighborsSearcher<F>` exposes `k_nearest` and `radius_neighbors` over borrowed
dense query views, returning owned dense result matrices. Both trees implement it; the
brute-force searcher added in W3.3 implements the same trait. This is a real abstraction
used by KNN and DBSCAN, not a placeholder: it lets the estimator choose an algorithm at
runtime without leaking the tree type into its model.

Alternatives considered: a private enum inside the estimator (would duplicate the query
surface and block DBSCAN reuse); free functions per tree (no shared type for the model).

### Decision 7 — Depth-first traversal with a bounded candidate list

The builder's `execution_mode` is carried by the index as configuration. Query batches choose
their parallel or sequential path from execution planning (`sk_should_run_parallel`) rather
than from a hard-coded flag; the configured mode is not consulted at query time, by design.

Each query keeps a sorted, bounded list of the current `k` best candidates and a LIFO stack of nodes
ordered by their lower bound. A node is expanded only when its bound is smaller than the
current worst candidate. Leaves are scanned with auto-vectorizable iterator loops and candidates are inserted
with a comparison on `(distance, index)`. Radius search uses the same traversal and pushes
every point within the radius into a per-row vector.

### Decision 8 — Deterministic ordering and parallel batches

Final results are sorted by `(distance, training_index)`, using `total_cmp` after
rejecting non-finite values. Query rows are independent, so the batch is evaluated with
`par_iter().map(...).collect()`, which preserves row order. The grain is a product of
query rows and training rows; below it the path stays sequential.

### Decision 9 — Builders, validation and observability

`SKKDTreeBuilder<F>` and `SKBallTreeBuilder<F>` expose `leaf_size` (default 40, matching
scikit-learn), `metric` (default Euclidean), `backend` (default `Faer`) and
`execution_mode` (default `Automatic`). `build` takes an `SKDataView`, validates the
representation and finiteness, records a `build` span with the backend and returns the
index. Queries record `k_nearest`/`radius_neighbors` spans. Errors use the central
taxonomy: `EmptyInput`, `InvalidHyperparameter`, `ShapeMismatch`, `NonFiniteValue`,
`UnsupportedRepresentation`.

### Decision 10 — File layout

```
crates/sciencekit_neighbors/src/
  lib.rs
  metric/{mod.rs, catalogue.rs, tolerance.rs, *_tests.rs}
  searcher/{mod.rs, contract.rs, query_result.rs, candidates.rs, validation.rs, *_tests.rs}
  kd_tree/{mod.rs, builder.rs, node.rs, build.rs, query/, *_tests.rs}
  ball_tree/{mod.rs, builder.rs, node.rs, build/, query/, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; every implementation file stays under 200 lines and
companion `*_tests.rs` modules are declared at the end of their dispatcher.

## Risks / Trade-offs

- **Curse of dimensionality in KD-trees** → KD is the default for low-dimensional data;
  BallTree is the documented alternative and the estimator's `Auto` policy (W3.3) can
  pick brute force when dimensions are high.
- **Node bounds cost `O(nodes · features)`** → document the memory profile; BallTree has
  the same order; no hidden densification of sparse input because sparse is rejected.
- **Floating-point ties** → deterministic `(distance, index)` ordering and exact equality
  for the tie case; tests include equidistant candidates.
- **`select_nth_unstable_by` reorders equal elements** → the comparator includes the
  training index, so the permutation is deterministic.
- **Per-query allocation** → reuse per-row buffers inside the parallel map; the result
  matrices are allocated once per batch.
- **`k` larger than the training set** → rejected rather than padded, so callers cannot
  silently consume sentinel rows.
- **Backend is carried but trees do no BLAS** → the field is still recorded in spans and
  transferred to downstream model configuration, preserving the uniform selection
  surface.

## Migration Plan

Additive: a new crate and new workspace membership. No existing API changes. Removing the
change deletes the crate; downstream W3.3/W3.4 code does not exist yet.

## Open Questions

None blocking. A sparse spatial index and an approximate index are separate future
changes; `memmap2` backing and persistence are Wave 7.
