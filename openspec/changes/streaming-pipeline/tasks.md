# Tasks

## 1. Contract and builder

- [x] 1.1 Create the `streaming` module with `SKStreamingTransformer<F>`, the builder and
  the empty-pipeline validation; verify `cargo test -p sciencekit_pipeline streaming`
  compiles.
- [x] 1.2 Implement a test streaming transformer that learns a running mean and
  transforms batches, with one-pass and two-epoch tests; verify with
  `cargo test -p sciencekit_pipeline streaming_transformer`.

## 2. Chained fitting

- [x] 2.1 Implement chained fitting over a reopenable provider and a two-stage test; verify
  with `cargo test -p sciencekit_pipeline streaming_fit`.
- [x] 2.2 Implement the one-shot provider restriction and the memory-budget error; verify
  with `cargo test -p sciencekit_pipeline streaming_oneshot`.
- [x] 2.3 Implement multi-epoch fitting and epoch reporting; verify with
  `cargo test -p sciencekit_pipeline streaming_epochs`.

## 3. Transform and finalizer

- [ ] 3.1 Implement per-batch transform through the fitted stages and compare it with the
  equivalent batch pipeline; verify with `cargo test -p sciencekit_pipeline streaming_transform`.
- [ ] 3.2 Add dense/sparse flow and no-densification tests; verify with
  `cargo test -p sciencekit_pipeline streaming_representation`.
- [ ] 3.3 Implement the bounded batch finalizer and the budget-exceeded error; verify with
  `cargo test -p sciencekit_pipeline streaming_finalizer`.
- [ ] 3.4 Add source-error and stage-error propagation tests; verify with
  `cargo test -p sciencekit_pipeline streaming_errors`.
- [ ] 3.5 Add repeated-fit determinism, concurrent transform and `Send + Sync` tests;
  verify with `cargo test -p sciencekit_pipeline streaming_concurrency`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large streams, one/multi-epoch
  providers, dense/sparse, chained transformers, finalizer budgets, error propagation and
  metric production; verify with `cargo test -p sciencekit_pipeline`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on streaming stages, epochs and finalizers); verify with
  `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `streaming-pipeline`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
