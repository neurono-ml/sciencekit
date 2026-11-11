# Tasks

## 1. Scheduler

- [x] 1.1 Implement the round scheduler and resource schedule, with the `9 → 3 → 1` and
  `10, 30, 90` tests; verify with
  `cargo test -p sciencekit_model_selection halving_scheduler`.
- [ ] 1.2 Implement deterministic candidate ranking, tie-breaking and the elimination
  trace; verify with `cargo test -p sciencekit_model_selection halving_trace`.

## 2. Subsample and evaluation

- [ ] 2.1 Implement seeded resource subsampling and stratified classification
  subsampling; verify with `cargo test -p sciencekit_model_selection halving_subsample`.
- [ ] 2.2 Reuse the grid-search evaluation over the subsampled folds and store round
  results; verify with `cargo test -p sciencekit_model_selection halving_evaluation`.
- [ ] 2.3 Implement aggressive elimination and test a round that prunes candidates early;
  verify with `cargo test -p sciencekit_model_selection halving_aggressive`.

## 3. Estimators and prediction

- [ ] 3.1 Implement the classifier/regressor builders with defaults and validation
  (`factor`, resources, empty candidates, incompatible scorer); verify with
  `cargo test -p sciencekit_model_selection halving_builder`.
- [ ] 3.2 Implement refit, best-model accessors and classifier/regressor prediction
  delegation; verify with `cargo test -p sciencekit_model_selection halving_refit`.
- [ ] 3.3 Add prediction-without-refit, dense/sparse and concurrency tests; verify with
  `cargo test -p sciencekit_model_selection halving_concurrency`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large data, classifier/regressor,
  factor variations, resource schedules, stratified/group splitters, aggressive mode,
  determinism and metric production; verify with `cargo test -p sciencekit_model_selection`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on successive halving, resources and elimination);
  verify with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `halving-grid-search`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
