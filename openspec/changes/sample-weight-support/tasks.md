# Tasks

Every task follows red-green-refactor: companion test first, confirmed failure, minimal
implementation, refactor with the gates green.

## 1. Contract and batches

- [ ] 1.1 Add the shared weight validation (`finite`, `non-negative`, length match) with a
  failing test per violation; verify with `cargo test -p sciencekit_common sample_weight`.
- [ ] 1.2 Extend `SKDataBatch` and `SKSupervisedBatch` with optional weights and the guard
  accounting; verify with `cargo test -p sciencekit_common batching`.
- [ ] 1.3 Replace the synthesized ones in SGD with real batch weights; verify with
  `cargo test -p sciencekit_linear_model sgd_weights`.

## 2. Estimators

- [ ] 2.1 Add weights to the aggregating scalers and the simple imputer; verify with
  `cargo test -p sciencekit_preprocessing -p sciencekit_impute weighted`.
- [ ] 2.2 Add weights to the linear and logistic families through the existing weighted
  accumulators; verify with `cargo test -p sciencekit_linear_model weighted`.
- [ ] 2.3 Add weights to the SVM family (box capacities) and the tree/ensemble families
  (impurity and bootstrap frequencies); verify with
  `cargo test -p sciencekit_svm -p sciencekit_tree -p sciencekit_ensemble weighted`.
- [ ] 2.4 Add weights to KMeans and MiniBatchKMeans (weighted centroids and inertia);
  verify with `cargo test -p sciencekit_cluster weighted`.
- [ ] 2.5 Add the typed refusal for estimators whose mathematics does not admit weights;
  verify with `cargo test --workspace weighted_refusal`.

## 3. Metrics and parity

- [ ] 3.1 Expose weighted scoring entry points on the scorers; verify with
  `cargo test -p sciencekit_metrics weighted`.
- [ ] 3.2 Prove "ones equal absent" and "zero rows vanish" per family; verify with
  `cargo test --workspace weighted`.
- [ ] 3.3 Prove weighted streaming equals weighted in-memory; verify with
  `cargo test --workspace weighted_streaming`.

## 4. Documentation

- [ ] 4.1 Add a Development Explained chapter on per-sample weights, the composition rule
  and the streaming batches; verify `mdbook build docs` and registration in `SUMMARY.md`.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
