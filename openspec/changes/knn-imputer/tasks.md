# Tasks

## 1. Module scaffold and marker semantics

- [ ] 1.1 Create `crates/sciencekit_impute/src/knn/` with dispatcher `mod.rs`, builder,
  config and tests; verify `cargo test -p sciencekit_impute knn` compiles.
- [ ] 1.2 Write and implement the missing-marker predicate (`NaN` via `is_nan`, other
  scalars via equality) and the dense missing mask, with tests for both markers; verify
  with `cargo test -p sciencekit_impute knn_marker`.

## 2. Fit state

- [ ] 2.1 Write `fit_tests.rs` for the retained training design, missing mask, valid
  columns (non-all-missing), observed column means and `keep_empty_features`; confirm the
  failures, then implement `fit.rs` and the model fields; verify with
  `cargo test -p sciencekit_impute knn_fit`.
- [ ] 2.2 Add backend/execution-mode transfer and span tests; verify with
  `cargo test -p sciencekit_impute knn_fit`.

## 3. Distance blocks and donor selection

- [ ] 3.1 Implement the chunked distance block over
  `sk_nan_euclidean_distance_matrix`, with tests for chunk boundaries, a single receiver
  and a large receiver count; verify with `cargo test -p sciencekit_impute knn_distance`.
- [ ] 3.2 Implement per-column donor indices (training rows observing the column) and the
  `k`-smallest partial selection, with tests for donors that miss the target column and
  for a zero potential-donor column; verify with
  `cargo test -p sciencekit_impute knn_donors`.
- [ ] 3.3 Implement the uniform and inverse-distance weighted average, the zero-distance
  convention and the column-mean fallback, with hand-computed tests; verify with
  `cargo test -p sciencekit_impute knn_average`.

## 4. Transform

- [ ] 4.1 Implement the dense transform returning the valid columns (or all columns with
  zeros when `keep_empty_features`), with tests for a known matrix and an all-missing
  column; verify with `cargo test -p sciencekit_impute knn_transform`.
- [ ] 4.2 Implement `add_indicator` composition using the fitted `SKMissingIndicator`
  scope, with tests for the column layout and missingness unseen at fit; verify with
  `cargo test -p sciencekit_impute knn_indicator`.
- [ ] 4.3 Add the bounded-memory chunk plan and the row-parallel path, with a test that
  the full distance matrix is not materialized and parallel equals sequential; verify with
  `cargo test -p sciencekit_impute knn_transform`.

## 5. Validation and dense-only contract

- [ ] 5.1 Add tests for empty input, zero/excessive neighbour count, feature-count
  mismatch, non-finite values with a non-NaN marker and sparse input rejection, asserting
  the exact central error variant; verify with
  `cargo test -p sciencekit_impute knn_validation`.
- [ ] 5.2 Add a `Send + Sync` assertion and a concurrent-transform test sharing one
  model; verify with `cargo test -p sciencekit_impute knn_concurrency`.

## 6. Acceptance

- [ ] 6.1 Add `acceptance_tests/` covering tiny data (one receiver, one donor), large data
  (at least 10 000 rows), every missing pattern, weighted/unweighted donors, mean fallback,
  indicator composition, concurrency and imputation error metrics through
  `sciencekit_metrics`; verify with `cargo test -p sciencekit_impute`.
- [ ] 6.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.

## 7. Documentation, changelog and review

- [ ] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for `SKKNNImputer`, the chunked `nan_euclidean` transform and the indicator
  composition.
- [ ] 7.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the KNN imputation chapter, register it in `SUMMARY.md`, add
  a `Development Explained` chapter covering `nan_euclidean`, donor selection, the
  fallback and the dense-only exception, update `maintaining.md`, and verify with
  `mdbook build docs`.
- [ ] 7.3 Record the non-metric `nan_euclidean` rationale, the dense-only exception and
  the deferred future index in `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 7.4 Independent review by a separate agent validating every scenario in
  `nearest-neighbor-imputation` against the code and tests; address every finding and
  re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
