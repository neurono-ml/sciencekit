# Tasks

## 1. Workspace and crate scaffold

- [x] 1.1 Create `crates/sciencekit_neighbors` with `Cargo.toml`, `lib.rs` and dispatcher
  `mod.rs` files for `metric`, `searcher`, `kd_tree` and `ball_tree`; add it to
  `[workspace.members]`; verify `cargo test -p sciencekit_neighbors` compiles an empty
  crate.
- [x] 1.2 Add the shared result types `SKNeighborQuery<F>` and `SKRadiusNeighbors<F>` with
  shape accessors, and verify their unit tests cover empty batches, zero neighbours and
  row-major indexing.

## 2. Metric catalogue

- [x] 2.1 Write `metric_tests.rs` for `SKNeighborMetric` point distances — Euclidean,
  squared Euclidean, Manhattan, Chebyshev, identical rows, symmetry and non-negativity —
  confirm the failure, then implement `metric.rs`; verify with
  `cargo test -p sciencekit_neighbors metric`.
- [x] 2.2 Test and implement the KD axis-aligned lower bound and the ball lower bound
  `max(0, d(q, centroid) − radius)`, including a query inside the ball (bound zero) and
  outside it; verify with `cargo test -p sciencekit_neighbors bound`.

## 3. Searcher contract

- [x] 3.1 Define `SKNearestNeighborsSearcher<F>` with `k_nearest` and
  `radius_neighbors` over borrowed dense query views; add a tiny mock implementation in
  `searcher_tests.rs` proving the contract is object-safe enough for the estimator's
  runtime enum; verify with `cargo test -p sciencekit_neighbors searcher`.
- [x] 3.2 Add the shared query validation helper (feature-count match, finite values,
  `k > 0`, `k <= training_rows`) with tests for every error variant; verify with
  `cargo test -p sciencekit_neighbors validation`.

## 4. KD-tree construction

- [x] 4.1 Write `kd_tree/build_tests.rs` for empty input, `leaf_size` zero, a single row,
  the permutation being a permutation, every leaf holding at most `leaf_size` rows and
  contiguous ranges; confirm the failures, then implement the node array and median
  split in `kd_tree/build.rs`; verify with `cargo test -p sciencekit_neighbors kd_build`.
- [x] 4.2 Add a test proving the build is deterministic for duplicated feature values and
  that the `(value, training_index)` comparator fixes the split; verify with
  `cargo test -p sciencekit_neighbors kd_build`.
- [x] 4.3 Add the build span with the configured backend and the public accessors
  (`leaf_size`, `metric`, `backend`, `training_rows`, `node_count`, `depth`); verify with
  `cargo test -p sciencekit_neighbors kd_build`.

## 5. KD-tree queries

- [x] 5.1 Write `kd_tree/query_tests.rs` for known 2-D nearest neighbours, a training row
  querying itself, `k = 1`, `k = n`, sorted distances and shape; confirm the failure, then
  implement best-first `k_nearest` in `kd_tree/query.rs`; verify with
  `cargo test -p sciencekit_neighbors kd_query`.
- [x] 5.2 Add deterministic tie tests with equidistant candidates and assert the smaller
  training index wins; verify with `cargo test -p sciencekit_neighbors kd_query`.
- [x] 5.3 Implement `radius_neighbors` with inclusive boundary, empty neighbourhoods and
  unsorted-then-sorted output; verify with `cargo test -p sciencekit_neighbors kd_query`.
- [x] 5.4 Add the row-parallel query path with a test proving parallel equals sequential
  on a large batch and an explicit-grain test; verify with
  `cargo test -p sciencekit_neighbors kd_query`.

## 6. Ball-tree construction and queries

- [x] 6.1 Write `ball_tree/build_tests.rs` for centroid/radius invariants (every member
  inside its node radius), leaf sizes, single row and determinism; confirm the failures,
  then implement the farthest-pivot split in `ball_tree/build.rs`; verify with
  `cargo test -p sciencekit_neighbors ball_build`.
- [x] 6.2 Write `ball_tree/query_tests.rs` for known neighbours, `k = n`, radius boundary,
  tie ordering and shape; confirm the failures, then implement best-first `k_nearest` and
  `radius_neighbors` in `ball_tree/query.rs`; verify with
  `cargo test -p sciencekit_neighbors ball_query`.
- [x] 6.3 Add the row-parallel ball query path and a large-batch equivalence test; verify
  with `cargo test -p sciencekit_neighbors ball_query`.

## 7. Builders, representation and errors

- [x] 7.1 Implement `SKKDTreeBuilder` and `SKBallTreeBuilder` with `leaf_size` default 40,
  `metric` default Euclidean, `backend` default `Faer`, `execution_mode` default
  `Automatic`, and `build` over `SKDataView`; add default-override tests; verify with
  `cargo test -p sciencekit_neighbors builder`.
- [x] 7.2 Add tests proving a sparse training view and a sparse query view return
  `UnsupportedRepresentation` with guidance, and that no sparse input is densified; verify
  with `cargo test -p sciencekit_neighbors representation`.
- [x] 7.3 Add tests proving cosine and NaN-aware Euclidean configurations return
  `InvalidHyperparameter`, and that NaN/infinity in training or queries returns
  `NonFiniteValue` at the known position; verify with
  `cargo test -p sciencekit_neighbors validation`.

## 8. Acceptance and integration

- [x] 8.1 Add `acceptance_tests/` covering tiny data (one row, one query), large data (at
  least 20 000 rows, 10 000 queries), every supported metric, both trees, radius queries,
  deterministic ties and concurrent query batches; verify with
  `cargo test -p sciencekit_neighbors`.
- [x] 8.2 Add a compile-time `Send + Sync` assertion for both index types and a
  concurrent shared-index test; verify with `cargo test -p sciencekit_neighbors`.

## 9. Documentation, changelog and review

- [x] 9.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the new crate, both indexes, the searcher contract and the metric catalogue;
  verify the entry lists every public artifact.
- [x] 9.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the neighbours chapter to `docs/src/`, register it in
  `SUMMARY.md`, add a `Development Explained` chapter covering KD/ball trees, bounds,
  deterministic ties and the dense-only index contract, update `maintaining.md`, and verify
  with `mdbook build docs`.
- [x] 9.3 Record the Wave 7 deferral in `docs/PRD.md` — `memmap2`-backed indexes,
  persistence and model export are delivered by the interop wave, not by this change —
  and verify the amended paragraphs render in the book.
- [x] 9.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 9.5 Independent review by a separate agent validating every scenario in
  `nearest-neighbor-search` against the code and tests; address every finding and re-run
  the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
