# Tasks

Every task follows the red-green-refactor loop: write the failing companion test first,
confirm the failure, implement the minimum that passes and refactor with the gates green.

## 1. Calibration curve

- [ ] 1.1 Add `SKCalibrationStrategy`, `SKCalibrationBin` and `SKCalibrationCurve` with
  construction tests for bin summaries and total sample count; verify with
  `cargo test -p sciencekit_metrics calibration_curve`.
- [ ] 1.2 Implement uniform binning over `[0, 1]` with the documented edge inclusion and
  retained empty bins, plus equal-width and empty-bin tests; verify with
  `cargo test -p sciencekit_metrics calibration_curve`.
- [ ] 1.3 Implement quantile binning with empirical edges and duplicate-edge collapsing,
  plus tied-score and single-distinct-value tests; verify with
  `cargo test -p sciencekit_metrics calibration_curve`.
- [ ] 1.4 Implement `sk_calibration_curve` validation for empty input, length mismatch,
  invalid bin count, invalid strategy and non-finite probabilities; verify with
  `cargo test -p sciencekit_metrics calibration_validation`.

## 2. Calibration metrics

- [ ] 2.1 Implement top-label reliability data over a multiclass probability matrix with
  deterministic argmax ties and correctness tests; verify with
  `cargo test -p sciencekit_metrics top_label`.
- [ ] 2.2 Implement `sk_brier_score` with perfect, constant-half and shape-rejection tests;
  verify with `cargo test -p sciencekit_metrics brier`.
- [ ] 2.3 Implement `sk_expected_calibration_error` over the shared bins with
  calibrated, miscalibrated and bounded-error fixtures; verify with
  `cargo test -p sciencekit_metrics expected_calibration_error`.
- [ ] 2.4 Add calibration parity fixtures comparing uniform and quantile curves and the
  matching expected calibration error, plus f32/f64 tolerance tests; verify with
  `cargo test -p sciencekit_metrics calibration_parity`.

## 3. Multilabel representation and losses

- [ ] 3.1 Add the dense/CSR indicator boundary with a single representation dispatch, the
  indicator-value validation and non-densification tests; verify with
  `cargo test -p sciencekit_metrics multilabel_representation`.
- [ ] 3.2 Implement per-row true-positive, false-positive and false-negative counting over
  dense and CSR rows, with dense/sparse equivalence tests; verify with
  `cargo test -p sciencekit_metrics multilabel_counts`.
- [ ] 3.3 Implement `sk_hamming_loss` and `sk_subset_accuracy` with perfect, single-flip
  and partially-correct fixtures; verify with `cargo test -p sciencekit_metrics multilabel_loss`.
- [ ] 3.4 Implement `sk_jaccard_score` with binary, micro, macro, weighted and samples
  averaging and the zero-division policy; verify with
  `cargo test -p sciencekit_metrics jaccard`.

## 4. Multilabel averaging and ranking

- [ ] 4.1 Add `SKMultilabelAverage`, `SKLabelScore` and `SKMultilabelReport` with
  construction and per-label completeness tests; verify with
  `cargo test -p sciencekit_metrics multilabel_report`.
- [ ] 4.2 Implement `sk_multilabel_precision_recall_f1` with micro pooling, macro mean,
  support weighting and samples averaging, with hand-computed fixtures per mode; verify
  with `cargo test -p sciencekit_metrics multilabel_precision_recall_f1`.
- [ ] 4.3 Add a scikit-learn parity fixture for every averaging mode, including absent
  labels and zero denominators; verify with
  `cargo test -p sciencekit_metrics multilabel_average_parity`.
- [ ] 4.4 Implement `sk_label_ranking_average_precision` with perfect-ranking, partial and
  empty-true-label rejection tests; verify with
  `cargo test -p sciencekit_metrics label_ranking_average_precision`.

## 5. Validation, acceptance and review

- [ ] 5.1 Add validation tests for empty input, shape and length mismatch, invalid
  indicator values and non-finite values across every new function; verify with
  `cargo test -p sciencekit_metrics multilabel_validation`.
- [ ] 5.2 Add large-data, f32/f64, concurrency and `Send + Sync` acceptance tests under
  `acceptance_tests/`; verify with `cargo test -p sciencekit_metrics`.
- [ ] 5.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 5.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on reliability diagrams, Brier score, expected
  calibration error and multilabel averaging); verify with `mdbook build docs`.
- [ ] 5.5 Independent review by a separate agent validating every scenario in
  `metrics-calibration-multilabel` and its scikit-learn parity claims; address every
  finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
