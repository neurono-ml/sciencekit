# Tasks

## 1. Probability scorer contract

- [x] 1.1 Add `SKProbabilityScorer<F, M>` to `sciencekit_common` with pure and convenient
  forms, and a constant-classifier test; verify with
  `cargo test -p sciencekit_common probability_scorer`.
- [x] 1.2 Add probability shape validation and tests; verify with
  `cargo test -p sciencekit_common probability_scorer`.

## 2. Ranking metrics

- [x] 2.1 Implement binary ROC curve and AUC with rank-sum tie handling, plus perfect,
  random and tied fixtures; verify with `cargo test -p sciencekit_metrics roc`.
- [x] 2.2 Implement multiclass OvR ROC/AUC with macro/micro/weighted averaging and
  per-class tests; verify with `cargo test -p sciencekit_metrics roc`.
- [x] 2.3 Implement precision-recall curves and average precision with step interpolation
  and perfect/empty-positive fixtures; verify with
  `cargo test -p sciencekit_metrics precision_recall`.
- [x] 2.4 Implement binary/multiclass log loss with clipping and finite-loss tests; verify
  with `cargo test -p sciencekit_metrics log_loss`.
- [x] 2.5 Implement balanced accuracy and Matthews correlation with hand-computed fixtures;
  verify with `cargo test -p sciencekit_metrics balanced_mcc`.

## 3. Scorers and accumulator

- [x] 3.1 Implement `SKAucScorer`, `SKAveragePrecisionScorer`, `SKLogLossScorer`,
  `SKBalancedAccuracyScorer` and `SKMatthewsCorrelationScorer`, with adapter-equals-free-
  function tests; verify with `cargo test -p sciencekit_metrics ranking_scorers`.
- [x] 3.2 Implement `SKAucAccumulator` with observe/merge/finalize and batched/merged
  equality tests; verify with `cargo test -p sciencekit_metrics auc_accumulator`.

## 4. Validation, acceptance and review

- [x] 4.1 Add validation tests for empty input, length mismatch, probability range,
  single-class AUC and non-finite values; verify with
  `cargo test -p sciencekit_metrics ranking_validation`.
- [x] 4.2 Add large-data, f32/f64, concurrency and `Send + Sync` acceptance tests; verify
  with `cargo test -p sciencekit_metrics`.
- [x] 4.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on ROC/AUC, average precision, log loss and MCC); verify
  with `mdbook build docs`.
- [x] 4.5 Independent review by a separate agent validating every scenario in
  `ranking-metrics` and the probability scorer delta; address every finding and re-run the
  gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
