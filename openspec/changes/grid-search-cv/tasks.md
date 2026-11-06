# Tasks

## 1. Scoring and configuration

- [ ] 1.1 Implement `SKClassifierScoring` and `SKRegressorScoring` enums dispatching to the
  metrics crate, with hand-computed values and pair/partition tests; verify with
  `cargo test -p sciencekit_model_selection grid_scoring`.
- [ ] 1.2 Implement the classifier/regressor builders with candidate, splitter, scoring,
  refit, error-score, groups and execution-mode setters; verify defaults and invalid values
  with `cargo test -p sciencekit_model_selection grid_builder`.

## 2. Evaluation

- [ ] 2.1 Implement candidate/fold evaluation with gathered train/test blocks, storing
  results by candidate and fold index; verify a hand-computed three-candidate/five-fold
  matrix; verify with `cargo test -p sciencekit_model_selection grid_evaluation`.
- [ ] 2.2 Implement mean/std aggregation, best selection and deterministic ties; verify
  with `cargo test -p sciencekit_model_selection grid_selection`.
- [ ] 2.3 Implement `error_score` Raise/Value behavior and a candidate that fails one fold;
  verify with `cargo test -p sciencekit_model_selection grid_errors`.
- [ ] 2.4 Implement refit and the best-model accessors; verify with
  `cargo test -p sciencekit_model_selection grid_refit`.

## 3. Prediction and representation

- [ ] 3.1 Implement classifier labels/probabilities and regressor responses delegating to
  the best model; verify with `cargo test -p sciencekit_model_selection grid_predict`.
- [ ] 3.2 Add dense/sparse fold-gathering tests with no densification and a sparse grid
  search; verify with `cargo test -p sciencekit_model_selection grid_representation`.
- [ ] 3.3 Add prediction-without-refit and concurrency tests; verify with
  `cargo test -p sciencekit_model_selection grid_concurrency`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large data, classifier/regressor,
  every scoring policy, multiple splitters, refit on/off, error scores, determinism and
  metric production; verify with `cargo test -p sciencekit_model_selection`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on grid evaluation, aggregation and refit); verify with
  `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `grid-search`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
