# Tasks

Every task follows red-green-refactor: companion test first, confirmed failure, minimal
implementation, refactor with the gates green.

## 1. Shared contracts

- [ ] 1.1 Add `sk_validate_mappable_source` and the shared `SKSharedMappableSource` handle
  with rejection and lifetime tests; verify with
  `cargo test -p sciencekit_common mappable`.
- [ ] 1.2 Add the parity harness (in-memory vs mapped) and a failing fixture for a
  divergence; verify with `cargo test -p sciencekit_common mappable_parity`.

## 2. Neighbors and imputation

- [ ] 2.1 Implement the mapped fit for the KNN classifier and regressor; verify with
  `cargo test -p sciencekit_neighbors mappable`.
- [ ] 2.2 Implement the mapped fit for the KD-tree and ball-tree builders; verify with
  `cargo test -p sciencekit_neighbors mappable_tree`.
- [ ] 2.3 Implement the mapped fit for the KNN imputer; verify with
  `cargo test -p sciencekit_impute mappable`.

## 3. Clustering, trees and ensembles

- [ ] 3.1 Implement the mapped fit for DBSCAN; verify with
  `cargo test -p sciencekit_cluster mappable`.
- [ ] 3.2 Implement the mapped fit for the decision tree classifier and regressor; verify
  with `cargo test -p sciencekit_tree mappable`.
- [ ] 3.3 Implement the mapped fit for random forests and bagging through the bootstrap
  gather; verify with `cargo test -p sciencekit_ensemble mappable`.
- [ ] 3.4 Implement the mapped fit for the isolation forest; verify with
  `cargo test -p sciencekit_outlier mappable`.

## 4. Parity and observability

- [ ] 4.1 Run the parity harness per estimator family on dense and CSR fixtures; verify
  with `cargo test --workspace mappable`.
- [ ] 4.2 Record the declared pattern and the mapped source kind in the operation span;
  verify with `cargo test -p sciencekit_common operation_attributes`.
- [ ] 4.3 Confirm the in-memory call sites and their tests are unchanged; verify with
  `cargo test --workspace`.

## 5. Documentation

- [ ] 5.1 Add a Development Explained chapter on fitting from a mapping without
  materializing, grounded in the parity harness tests; verify `mdbook build docs` and
  registration in `SUMMARY.md`.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
