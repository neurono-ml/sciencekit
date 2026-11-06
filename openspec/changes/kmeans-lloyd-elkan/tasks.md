# Tasks

## 1. Crate and data path

- [ ] 1.1 Create `crates/sciencekit_cluster` with `Cargo.toml`, `lib.rs` and the `kmeans`
  dispatcher; add it to `[workspace.members]`; verify
  `cargo test -p sciencekit_cluster` compiles.
- [ ] 1.2 Implement the dense squared-Euclidean assignment kernel and tests for a
  hand-computed center set; verify with `cargo test -p sciencekit_cluster kmeans_distance`.
- [ ] 1.3 Implement the sparse norm-expansion assignment kernel and dense-vs-sparse
  equivalence tests, asserting no densification; verify with
  `cargo test -p sciencekit_cluster kmeans_distance`.

## 2. Initialization

- [ ] 2.1 Implement seeded uniform and k-means++ initialization with distinct-center and
  reproducibility tests; verify with `cargo test -p sciencekit_cluster kmeans_init`.
- [ ] 2.2 Implement `number_of_initializations` with parallel runs and lowest-inertia
  selection, including the tie-break test; verify with
  `cargo test -p sciencekit_cluster kmeans_init`.

## 3. Lloyd and Elkan

- [ ] 3.1 Implement Lloyd assignment/update with the convergence criterion, empty-cluster
  repair and iteration/convergence reporting; verify a hand-computed two-cluster fixture;
  verify with `cargo test -p sciencekit_cluster kmeans_lloyd`.
- [ ] 3.2 Implement Elkan center/row bounds and the pruning tests, and prove Elkan equals
  Lloyd on dense data within tolerance; verify with
  `cargo test -p sciencekit_cluster kmeans_elkan`.
- [ ] 3.3 Implement `Auto` resolving Elkan for dense and Lloyd for sparse, with tests for
  the reported algorithm and identical results; verify with
  `cargo test -p sciencekit_cluster kmeans_algorithm`.

## 4. Estimator and model

- [ ] 4.1 Implement `SKKMeansBuilder` with the documented defaults and invalid-value tests;
  verify with `cargo test -p sciencekit_cluster kmeans_builder`.
- [ ] 4.2 Implement `SKUnsupervisedFit` returning `SKKMeansModel` with centers, labels,
  inertia, iterations, convergence and backend; verify with
  `cargo test -p sciencekit_cluster kmeans_fit`.
- [ ] 4.3 Implement `SKClustererPredictor` assignments, `transform` distances and
  `fit_predict`; verify shapes, nearest-center semantics and scalar independence; verify
  with `cargo test -p sciencekit_cluster kmeans_predict`.
- [ ] 4.4 Add validation tests for empty input, zero/excessive clusters, non-finite values,
  invalid parameters and feature-count mismatch; verify with
  `cargo test -p sciencekit_cluster kmeans_validation`.

## 5. Acceptance and review

- [ ] 5.1 Add `acceptance_tests/` covering tiny data, large data (at least 50 000 rows),
  dense/sparse, Lloyd/Elkan/Auto, multiple initializations, determinism, concurrency and
  inertia/metric production through `sciencekit_metrics`; verify with
  `cargo test -p sciencekit_cluster`.
- [ ] 5.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 5.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on k-means++, Lloyd, Elkan and empty-cluster repair);
  verify with `mdbook build docs`.
- [ ] 5.4 Record the streaming/GPU/memmap/export non-goals in `docs/PRD.md`.
- [ ] 5.5 Independent review by a separate agent validating every scenario in `k-means`;
  address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
