# Tasks

## 1. Union core

- [x] 1.1 Create the `feature_union` module, the tuple `SKFeatureUnionStages` trait and a
  two-member scalar chain; verify `cargo test -p sciencekit_pipeline union` compiles.
- [x] 1.2 Implement fit/transform over the tuple and a hand-checkable two-member
  concatenation; verify with `cargo test -p sciencekit_pipeline union_fit`.
- [x] 1.3 Implement dense and sparse concatenation, including the mixed policy and column
  offsets, with dense/sparse/mixed tests; verify with
  `cargo test -p sciencekit_pipeline union_concat`.

## 2. Boxed fallback and errors

- [x] 2.1 Implement the boxed dynamic union and a test that it matches the tuple union;
  verify with `cargo test -p sciencekit_pipeline union_boxed`.
- [x] 2.2 Implement member-index error wrapping, row-count mismatch and empty-union
  validation; verify with `cargo test -p sciencekit_pipeline union_validation`.
- [ ] 2.3 Add accessors (member count, per-member widths, total width) and a concurrency
  test; verify with `cargo test -p sciencekit_pipeline union_model`.

## 3. Pipeline integration and acceptance

- [ ] 3.1 Use the union as a stage inside `SKPipeline` and test a scaler∪encoder→classifier
  chain; verify with `cargo test -p sciencekit_pipeline union_pipeline`.
- [ ] 3.2 Add `acceptance_tests/` covering tiny and large data, dense/sparse/mixed
  members, tuple/boxed construction, nested pipelines, concurrency and metric production;
  verify with `cargo test -p sciencekit_pipeline`.
- [ ] 3.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on parallel feature branches and concatenation); verify
  with `mdbook build docs`.
- [ ] 3.5 Independent review by a separate agent validating every scenario in
  `feature-union`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
