# Tasks

## 1. Graph construction

- [x] 1.1 Add `petgraph` to the workspace and create the `graph` module with
  `SKPipelineGraphBuilder`, node ids and edge insertion; verify `cargo test -p
  sciencekit_pipeline graph` compiles.
- [x] 1.2 Implement cycle, final-node and reachability validation with doomed-graph
  fixtures; verify with `cargo test -p sciencekit_pipeline graph_validate`.

## 2. Execution

- [x] 2.1 Implement deterministic topological order and per-node block storage; verify with
  `cargo test -p sciencekit_pipeline graph_order`.
- [x] 2.2 Implement fit over a diamond graph and compare it with the equivalent linear
  chain; verify with `cargo test -p sciencekit_pipeline graph_fit`.
- [x] 2.3 Implement prediction through the fitted graph and classifier/regressor wrappers;
  verify with `cargo test -p sciencekit_pipeline graph_predict`.
- [x] 2.4 Implement merge nodes and document/test edge-order concatenation; verify with
  `cargo test -p sciencekit_pipeline graph_merge`.

## 3. Representation, concurrency and acceptance

- [x] 3.1 Add dense/sparse flow tests, a representation rejection with node id and no
  implicit densification; verify with `cargo test -p sciencekit_pipeline graph_representation`.
- [x] 3.2 Add repeated-fit determinism, concurrent prediction and `Send + Sync` tests;
  verify with `cargo test -p sciencekit_pipeline graph_concurrency`.
- [ ] 3.3 Add `acceptance_tests/` covering tiny and large graphs, linear-equivalent graphs,
  disconnected/cyclic fixtures, dense/sparse, concurrency and metric production; verify
  with `cargo test -p sciencekit_pipeline`.
- [ ] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on DAG validation, topological execution and merges);
  verify with `mdbook build docs`.
- [ ] 3.6 Independent review by a separate agent validating every scenario in
  `pipeline-graph`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
