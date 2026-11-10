# Tasks

## 1. Dispatch surface

- [x] 1.1 Write failing tests for `SKPairwiseMetric` and
  `sk_pairwise_distance_matrix`: metric selection calls the matching kernel, a feature-count
  mismatch returns `ShapeMismatch`, empty inputs produce empty matrices, and a sparse
  `NanEuclidean` request returns `UnsupportedRepresentation`; verify with
  `cargo test -p sciencekit_math sparse_pairwise`.
- [x] 1.2 Add `pairwise/dispatcher.rs` with the enum and the dispatch skeleton, re-exported
  from `pairwise/mod.rs`; verify the task 1.1 tests pass.

## 2. Sparse row helpers

- [x] 2.1 Write tests for a sorted-row merge helper (`shared`, left-only, right-only
  coordinates, empty rows, duplicate-free canonical rows), confirm the failure, then
  implement it in `pairwise/sparse/row_merge.rs`; verify with
  `cargo test -p sciencekit_math sparse_`.

## 3. Sparse × sparse kernels

- [x] 3.1 Implement `sk_sparse_squared_euclidean_distance_matrix` and
  `sk_sparse_euclidean_distance_matrix` from stored values (norms plus shared-coordinate
  dot product), with tests proving equivalence to the dense kernels, empty rows, symmetry
  and zero diagonal; verify with `cargo test -p sciencekit_math sparse_`.
- [x] 3.2 Implement `sk_sparse_manhattan_distance_matrix` with shared/one-sided
  contributions and tests for dense equivalence and one-sided absolute values; verify with
  `cargo test -p sciencekit_math sparse_`.
- [x] 3.3 Implement `sk_sparse_chebyshev_distance_matrix` with a merge-based maximum and
  tests for dense equivalence, one-sided contributions and identical rows; verify with
  `cargo test -p sciencekit_math sparse_`.
- [x] 3.4 Implement `sk_sparse_cosine_distance_matrix` from the stored dot and norms, with
  tests for dense equivalence, scale invariance and the zero-norm policy; verify with
  `cargo test -p sciencekit_math sparse_`.

## 4. Sparse × dense and dense × sparse

- [x] 4.1 Add the sparse×dense and dense×sparse paths for every metric, with tests that
  compare each mixed computation against the dense kernel on the same logical matrix; verify
  with `cargo test -p sciencekit_math sparse_`.
- [x] 4.2 Add a test with a wide sparse matrix (many features, few stored entries) proving
  the computation succeeds without densifying, and a test for a one-sided dense feature
  maximum in Chebyshev; verify with `cargo test -p sciencekit_math sparse_`.

## 5. Parallelism and concurrency

- [x] 5.1 Add the rayon row-parallel path to the sparse kernels with the dense grain, and a
  test that parallel equals sequential on a large sparse input; verify with
  `cargo test -p sciencekit_math sparse_`.
- [x] 5.2 Add a concurrency test spawning several threads over the same sparse input and
  asserting deterministic equality with the single-threaded result; verify with
  `cargo test -p sciencekit_math sparse_`.

## 6. Dispatcher integration and acceptance

- [x] 6.1 Route the dispatcher through the specialized kernels for all four representation
  combinations and all metrics, with tests covering each cell of the metric × representation
  matrix; verify with `cargo test -p sciencekit_math sparse_`.
- [x] 6.2 Add `crates/sciencekit_math/src/pairwise/sparse/acceptance_tests/` covering tiny
  data (empty rows, a single stored entry), large data (at least 10 000 rows), every metric,
  mixed representations and concurrent execution, asserting the documented tolerances; verify
  with `cargo test -p sciencekit_math`.

## 7. Gates

- [x] 7.1 Verify `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` are
  green before any push.

## 8. Documentation, changelog and review

- [x] 8.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased → Added`
  for the sparse kernels, the dispatcher and the dense-only NaN-aware decision; verify every
  public artifact is listed.
- [x] 8.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): document the sparse pairwise path and the dispatcher in
  `docs/src/math-kernel.md`, add a `Development Explained` chapter on sparse distance
  computation (algorithms, dispatch, no-densification and the mixed-representation paths),
  register it in `SUMMARY.md`, update `maintaining.md`, and verify with `mdbook build docs`.
- [ ] 8.3 Independent review by a separate agent validating every `pairwise-distances`
  scenario and the design decisions; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR (Y-statement) in English and reference it from
  both PRs with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
