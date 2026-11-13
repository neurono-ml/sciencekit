# Proposal

## Why

`SKKNeighborsClassifier`, `SKKNeighborsRegressor` and `SKKNNImputer` all require exact
nearest-neighbor search, and today the only way to obtain neighbours is to compare every
query row against every training row through the pairwise-distance kernels. That is
`O(n·m)` per query batch and makes the neighbours block of Wave 3 unusable on the large
datasets required by the PRD's acceptance rule. `SKDBSCAN` (Wave 5) also specifies
`SKKDTree` as its region-query engine, so the spatial index is shared foundation, not a
KNN private detail.

## What Changes

- New sub-crate `sciencekit_neighbors` with two immutable spatial indexes: `SKKDTree`
  and `SKBallTree`.
- Builder-only construction (`SKKDTreeBuilder` / `SKBallTreeBuilder`) exposing
  `leaf_size`, `metric`, `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`); direct constructors stay private.
- A shared `SKNearestNeighborsSearcher<F>` contract with exact `k_nearest` and
  `radius_neighbors` queries, implemented by both trees. The brute-force searcher joins
  the same contract in `knn-classifier-regressor` (W3.3).
- Array-based node storage (start/end, left/right child, radius, bounds), median or
  farthest-point splits, auto-vectorizable leaf scans and rayon work across query rows.
- Deterministic query ordering: distances ascending, ties broken by the smallest training
  row index, never by hash-map iteration order.
- Dense-only index contract: tree geometry (axis-aligned boxes and ball centres/radii) is
  dense by construction, so a sparse training or query view is rejected with
  `UnsupportedRepresentation` and a suggestion to use the brute-force searcher added in
  W3.3. The *neighbours estimators* will accept dense, sparse and mixed inputs by
  dispatching between trees and brute force; this change supplies the dense index half.
- Tree metric catalogue: Euclidean, squared Euclidean, Manhattan and Chebyshev. Cosine
  distance and NaN-aware Euclidean are rejected because they are not metric-compatible
  with tree pruning (cosine distance violates the triangle inequality; sparse NaN has no
  missing marker).
- Validation errors for an empty training set, zero `leaf_size`, zero `k`, `k` greater
  than the training-row count, feature-count mismatch and non-finite training values.
- Explicit non-goals, recorded in the PRD: `memmap2`-backed indexes, persistence and
  model export are Wave 7 (interop) work, not this change.
- Companion `*_tests.rs` modules covering tiny and large data, deterministic ties,
  concurrent queries and every metric.

No estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `nearest-neighbor-search`: exact spatial-index construction and query behavior —
  supported metrics, the dense-only index contract, deterministic ordering, validation and
  concurrent query semantics.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_neighbors` workspace member; root `Cargo.toml`
  membership. No existing crate is modified except the workspace manifest.
- **APIs:** additive. `SKKDTree`, `SKBallTree`, their builders,
  `SKNearestNeighborsSearcher`, `SKNeighborQuery` and `SKRadiusNeighbors`.
- **Dependencies:** `ndarray`, `sprs`, `rayon`, `sciencekit_common`, `sciencekit_math`;
  no new external crate.
- **Downstream:** unblocks `SKKNeighborsClassifier` / `SKKNeighborsRegressor` (W3.3),
  the dense path of `SKKNNImputer` (W3.4) and `SKDBSCAN` (W5.3).
- **Acceptance criteria (PRD §8.7 / §10.3):** correct results on small and large data,
  safe concurrent queries, companion `*_tests.rs` modules, evaluation exercised in the
  downstream estimator changes through `sciencekit_metrics`. Model export is deferred to
  Wave 7 and that deferral is recorded in the PRD by the associated documentation change.
