# Tasks

## 1. Search algorithm and weights

- [ ] 1.1 Write tests for `SKNeighborsAlgorithm` resolution — `Auto` picks a tree for
  dense Euclidean/Manhattan/Chebyshev above the threshold, brute force for sparse,
  mixed and cosine, and an explicit incompatible pair returns
  `UnsupportedRepresentation`; confirm the failures, then implement `algorithm.rs`;
  verify with `cargo test -p sciencekit_neighbors algorithm`.
- [ ] 1.2 Write and implement `SKNeighborsWeights` (`Uniform`, `Distance`), including the
  zero-distance convention (only zero-distance neighbours contribute) and a
  hand-computed weighted average; verify with `cargo test -p sciencekit_neighbors weights`.
- [ ] 1.3 Implement the brute-force searcher over the mixed distance dispatcher with
  chunked query rows, `select_nth_unstable_by` for the `k` smallest `(distance, index)`
  pairs and deterministic sorting; verify against the tree searcher on the same dense data
  and against a hand-computed sparse example; verify with
  `cargo test -p sciencekit_neighbors brute_force`.

## 2. Classifier

- [ ] 2.1 Implement `SKKNeighborsClassifierBuilder` with defaults (`k = 5`, `Uniform`,
  `Auto`, `Euclidean`, leaf size 40, `Faer`, `Automatic`) and invalid-value tests; verify
  with `cargo test -p sciencekit_neighbors classifier_builder`.
- [ ] 2.2 Implement `fit` storing the training design in its original representation, the
  canonical class table and the optional tree index; verify dense tree-backed and sparse
  brute-force models report the expected strategy; verify with
  `cargo test -p sciencekit_neighbors classifier_fit`.
- [ ] 2.3 Implement `predict_labels` with weighted majority vote and smallest-class-index
  tie-breaking; verify a hand-computed fixture and a vote tie; verify with
  `cargo test -p sciencekit_neighbors classifier_predict`.
- [ ] 2.4 Implement `predict_probabilities` as weighted neighbour fractions over the full
  class table, with rows summing to one and zero-probability classes preserved; verify with
  `cargo test -p sciencekit_neighbors classifier_probabilities`.
- [ ] 2.5 Add nominal-target canonicalization tests and verify predictions are returned as
  canonical indices with the table available on the model; verify with
  `cargo test -p sciencekit_neighbors classifier_targets`.

## 3. Regressor

- [ ] 3.1 Implement `SKKNeighborsRegressorBuilder` with the same search/weights surface
  and defaults; verify with `cargo test -p sciencekit_neighbors regressor_builder`.
- [ ] 3.2 Implement `fit` and `predict` with uniform mean, distance-weighted mean and the
  zero-distance convention; verify hand-computed fixtures; verify with
  `cargo test -p sciencekit_neighbors regressor`.

## 4. Representation parity

- [ ] 4.1 Add dense-vs-sparse fit/predict equivalence tests for both estimators, asserting
  predictions agree within tolerance and no sparse input is densified; verify with
  `cargo test -p sciencekit_neighbors representation`.
- [ ] 4.2 Add mixed prediction tests (dense model on sparse query and sparse model on dense
  query) through the brute-force path; verify with
  `cargo test -p sciencekit_neighbors representation`.
- [ ] 4.3 Add a test that `Auto` on sparse data reports brute force and that an explicit
  `KdTree`/`BallTree` on sparse data returns `UnsupportedRepresentation`; verify with
  `cargo test -p sciencekit_neighbors representation`.

## 5. Validation and errors

- [ ] 5.1 Add tests for empty input, zero `k`, `k` larger than the training set, feature
  count mismatch, non-finite values, sparse NaN and unsupported metric/algorithm pairs,
  asserting the exact central error variant; verify with
  `cargo test -p sciencekit_neighbors validation`.
- [ ] 5.2 Add a compile-time `Send + Sync` assertion for both fitted models and a
  concurrent-prediction test sharing one model across threads; verify with
  `cargo test -p sciencekit_neighbors concurrency`.

## 6. Acceptance

- [ ] 6.1 Add `acceptance_tests/` covering tiny data (one row), large data (at least
  20 000 rows), every algorithm/metric/weights combination, deterministic ties,
  parallel-equals-sequential prediction and metric production through
  `sciencekit_metrics`; verify with `cargo test -p sciencekit_neighbors`.
- [ ] 6.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.

## 7. Documentation, changelog and review

- [ ] 7.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for both estimators, the runtime algorithm selection and the brute-force searcher.
- [ ] 7.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the neighbours chapter, register it in `SUMMARY.md`, add a
  `Development Explained` chapter covering voting, weights, the `Auto` policy and the
  representation dispatch, update `maintaining.md`, and verify with `mdbook build docs`.
- [ ] 7.3 Record the deferred `partial_fit`/`fit_streaming`, `memmap2` and export work in
  `docs/PRD.md`; verify the amended paragraphs render.
- [ ] 7.4 Independent review by a separate agent validating every scenario in
  `nearest-neighbor-estimators` against the code and tests; address every finding and
  re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
