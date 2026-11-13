# Tasks

## 1. Audit and matrix

- [x] 1.1 Enumerate every `fit_streaming`/`partial_fit` implementation and record its
  dense/sparse behavior in the coverage matrix; verify with a workspace test that every
  implementation is listed.
- [x] 1.2 Add a compile/test guard that a dense-only streaming estimator rejects CSR with
  `UnsupportedRepresentation`; verify with `cargo test --workspace sparse_streaming_guard`.

## 2. Preprocessing and impute

- [x] 2.1 Add sparse streaming tests for `SKStandardScaler`, `SKMinMaxScaler` and
  `SKRobustScaler`, fixing any `as_dense()` path; verify with
  `cargo test -p sciencekit_preprocessing sparse_streaming`.
- [x] 2.2 Add sparse streaming tests for the encoders and polynomial features; verify with
  `cargo test -p sciencekit_preprocessing sparse_streaming`.
- [x] 2.3 Add sparse streaming tests for `SKSimpleImputer` and `SKMissingIndicator`; verify
  with `cargo test -p sciencekit_impute sparse_streaming`.

## 3. Linear models

- [x] 3.1 Add sparse streaming tests for linear regression/ridge and fix any dense
  conversion; verify with `cargo test -p sciencekit_linear_model sparse_streaming`.
- [x] 3.2 Add sparse streaming tests for lasso/elastic-net/logistic and SGD; verify with
  `cargo test -p sciencekit_linear_model sparse_streaming`.

## 4. Acceptance and review

- [x] 4.1 Add multi-batch, multi-epoch, large-sparse, no-densification, concurrency and
  memory-bound acceptance tests; verify with `cargo test --workspace sparse_streaming`.
- [x] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on the coverage matrix and no-densification rule);
  verify with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `sparse-streaming-coverage`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
