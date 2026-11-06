# Tasks

## 1. Module and region query

- [ ] 1.1 Create `crates/sciencekit_cluster/src/dbscan/` with dispatcher `mod.rs`,
  config and region-query modules; verify `cargo test -p sciencekit_cluster dbscan`
  compiles.
- [ ] 1.2 Implement `Auto` resolution (tree vs brute force), including the sparse/mixed
  rejection of tree-only metrics, with tests for the reported strategy; verify with
  `cargo test -p sciencekit_cluster dbscan_query`.
- [ ] 1.3 Implement the radius-neighbour query through the shared searcher and a test that
  the point itself is included; verify with `cargo test -p sciencekit_cluster dbscan_query`.

## 2. Expansion and labeling

- [ ] 2.1 Implement core detection and cluster expansion on a hand-checkable fixture
  (core, border and noise points); verify with `cargo test -p sciencekit_cluster dbscan_fit`.
- [ ] 2.2 Implement deterministic cluster-id assignment in training-index order and test
  repeated fits; verify with `cargo test -p sciencekit_cluster dbscan_fit`.
- [ ] 2.3 Implement dense tree vs brute-force equality and sparse/dense equality tests,
  asserting no densification; verify with
  `cargo test -p sciencekit_cluster dbscan_representation`.

## 3. Estimator and assignment

- [ ] 3.1 Implement `SKDBSCANBuilder` with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_cluster dbscan_builder`.
- [ ] 3.2 Implement `SKUnsupervisedFit` returning the model with typed assignments, core
  indices, cluster count and `eps`; verify with `cargo test -p sciencekit_cluster
  dbscan_model`.
- [ ] 3.3 Implement `SKClustererPredictor` via the nearest core sample within `eps`, with
  tests for a nearby query (cluster variant) and a distant query (`Noise`); verify with
  `cargo test -p sciencekit_cluster dbscan_predict`.
- [ ] 3.4 Add validation tests for empty input, non-positive `eps`, zero
  `minimum_samples`, feature-count mismatch, non-finite values and unsupported
  metric/algorithm pairs; verify with `cargo test -p sciencekit_cluster dbscan_validation`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny data, large data (at least 20 000 rows),
  dense/sparse, tree/brute-force, deterministic labels, concurrency and metric production;
  verify with `cargo test -p sciencekit_cluster`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on density reachability, core/border/noise and the
  nearest-core prediction extension); verify with `mdbook build docs`.
- [ ] 4.4 Record the HDBSCAN/OPTICS/GPU/memmap/export non-goals in `docs/PRD.md`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in `dbscan`;
  address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
