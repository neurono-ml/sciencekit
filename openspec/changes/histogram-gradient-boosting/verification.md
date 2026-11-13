# Independent Review — Histogram Gradient Boosting

Task 7.4 verification of the `histogram-gradient-boosting` change.

- Code branch: `feat/histogram-gradient-boosting` (PR #361)
- OpenSpec branch: `feat/histogram-gradient-boosting-openspec` (PR #234)
- Documentation branch: `docs/histogram-gradient-boosting-book` (PR #362)
- Reviewer: separate agent, did not write the implementation.
- Gates (local, `CARGO_TARGET_DIR=/home/claudino/.cache/sciencekit-target-histogram-gb`,
  `CARGO_PROFILE_DEV_DEBUG=0`): `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p sciencekit_ensemble`, `cargo test --workspace` — all green (see §5).

## 1. Scenario matrix

Every scenario from `specs/histogram-gradient-boosting/spec.md` against the concrete
test that proves it.

| # | Requirement / scenario | Proving test | Verdict |
|---|---|---|---|
| 1 | Builder-only construction: default configuration | `hist_classifier_defaults_match_the_documented_configuration`, `hist_regressor_defaults_and_loss_selector` | PASS |
| 2 | Quantile binning: bins respect the maximum | `hist_binning_derives_at_most_maximum_bins` | PASS |
| 3 | Quantile binning: missing values get the missing bin | `hist_binning_routes_missing_to_dedicated_bin` | PASS |
| 4 | Quantile binning: binning is deterministic | `hist_binning_is_deterministic` (identical edges + missing bin on refit) | PASS |
| 5 | Split finding: histogram gain matches a hand-computed split | `hist_gain_matches_hand_computed_split` (gain `4.0`, boundary 1, `default_left=false`) | PASS |
| 6 | Split finding: constraint blocks a split | `hist_gain_rejects_splits_below_minimum_samples_leaf` | PASS |
| 7 | Split finding: leaf-wise growth respects the node budget | `hist_growth_respects_the_maximum_leaf_budget` (now asserts exactly 3 leaves / 5 nodes) + `hist_growth_splits_the_leaf_with_the_largest_own_gain` | PASS |
| 8 | Subtraction equals direct accumulation | `hist_subtract_equals_direct_accumulation` (counts, gradient and Hessian sums per bin within `1e-12`) | PASS |
| 9 | Missing value follows the learned direction | `hist_missing_learns_the_left_direction`, `hist_missing_learns_the_right_direction` | PASS |
| 10 | Missing routing is deterministic | `hist_missing_without_missing_rows_uses_deterministic_direction`, `hist_determinism_repeated_fits_agree` | PASS |
| 11 | Categories are partitioned, not ordered | `hist_categorical_partitions_categories_by_gradient_ratio` (mask `{0,2}`, gain `98/3`) | PASS |
| 12 | Unknown categorical value at prediction | `hist_growth_routes_an_unknown_category_away_from_the_left_mask`, `hist_acceptance_categorical_and_missing_values` | PASS (was WEAK, test added) |
| 13 | Early stop reports the fitted iteration count | `hist_early_stop_reports_the_fitted_iteration_count`, `hist_early_stop_keeps_every_stage_when_no_validation_split_is_possible` | PASS (was FAIL, defect 3 fixed) |
| 14 | Learning rate scales the leaf | `hist_stage_applies_a_hand_computed_newton_update` (`0.5 ± 0.1·0.5` → `0.45/0.55`) | PASS |
| 15 | Dense and sparse queries agree | `hist_representation_dense_and_sparse_queries_agree` | PASS |
| 16 | Sparse query is not densified | `hist_representation_dense_and_sparse_queries_agree` + `hist_dataset_does_not_densify_sparse_input` + code inspection (`sk_sparse_entry`, `outer_view`) | PASS |
| 17 | Sparse fit equals dense fit | `hist_representation_sparse_fit_equals_dense_fit` (`1e-12`) | PASS |
| 18 | Invalid bin count is rejected | `hist_classifier_rejects_invalid_parameters_and_single_class_targets`, `hist_dataset_rejects_out_of_range_maximum_bins` | PASS (dataset guard added) |
| 19 | Unknown categorical column is rejected | `hist_validation_accepts_missing_values_and_rejects_unknown_categorical_columns` | PASS |
| 20 | Single-class classification is rejected | `hist_classifier_rejects_invalid_parameters_and_single_class_targets` | PASS |
| 21 | Repeated fits agree | `hist_determinism_repeated_fits_agree` | PASS |
| 22 | Concurrent prediction agrees | `hist_determinism_concurrent_prediction_agrees`, `hist_determinism_model_is_send_and_sync`, `hist_determinism_is_thread_count_invariant` | PASS |

## 2. Convention audit (`AGENTS.md`)

| Rule | Verdict | Evidence |
|---|---|---|
| `SK` prefix on structs/traits, `sk_` on free functions and modules | PASS | Public items are `SK*` / `sk_*`; methods carry no prefix. Public modules (`binning`, `growth`, `histogram`) are unprefixed like the rest of the crate (`bagging`, `gradient_boosting`), matching repository practice. |
| No abbreviations in public names | WEAK (accepted) | `SKHistGradientBoosting*` shortens “Histogram” and `n_iter_no_change` uses `n`/`iter`. Both are inherited from the Wave-4 planning proposals (`decision-tree-cart`, `gradient-boosting`) and mirror the scikit-learn surface; renaming would contradict the pre-approved plan and the docs PR, so it is recorded as planning debt rather than changed mid-review. |
| Builder-only construction, private constructors | PASS | Estimators have no public constructor; `SKBuilder::build` validates the configuration. |
| `backend(SKBackendKind)` defaults to `Faer`; recorded in every operation span | PASS | `SKHistogramGradientBoostingConfiguration::new` sets `Faer`; `fit`, `probabilities` and `predict_values` build `SKOperationAttributes { backend: … }` from the estimator/model kind. |
| Dense/sparse parity, never densify | PASS | CSR is binned through its stored entries plus implicit zero; prediction reads `sk_sparse_entry`. CSC is rejected with `UnsupportedRepresentation`. |
| Folder modules with a pure-dispatcher `mod.rs` | PASS | `mod.rs` files only declare submodules, re-export and register `*_tests` at the end. |
| Companion `*_tests.rs` at the end of `mod.rs` | PASS | All test modules are companion files declared last. |
| No manual index loops | PASS | Histogram and growth loops use iterators/`for_each`; `(0..n).for_each` only appears in the column-major scatter of `dataset.rs` and the prefix scan of `gain.rs`, where an index is the iteration state. |
| Files at or under 200 lines | WEAK (accepted) | `growth/leaf_wise.rs` (276) and `binning/dataset.rs` (252) exceed 200. The repository already ships larger implementation files (`sciencekit_math/src/backend/faer_backend.rs` 389), and splitting a working algorithm would be pure restructuring, so it is recorded instead of changed during review. |

## 3. Numerical honesty

- Tests assert hand-checkable numbers rather than “does not panic”: gain `4.0`, categorical
  gain `98/3`, Newton leaf values `±1` and `0.45/0.55`, `λ` cases `1.0` and `1/3`,
  subtraction identity per bin within `1e-12`, sparse/dense predictions within `1e-12`.
- The histogram splitter uses the same Newton gain as the exact splitter
  (`G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`), but there is no cross-check test against the
  exact `sciencekit_tree` splitter and no documented numeric tolerance for histogram-vs-exact
  disagreement. This is by design (binning changes candidate boundaries; the exact splitter
  remains available), yet the design’s “bins lose precision → documented” claim has no
  tolerance figure. Recorded as an observation, not a spec scenario.
- Silent fallbacks: the early-stopping truncation without a validation split was one and is
  fixed (§4). The remaining ones are explicit and documented: categorical codes beyond
  `maximum_bins − 1` share the unknown bin, and an impossible validation split degrades to
  fitting every row/stage.

## 4. Defects found and fixed (TDD: failing test first)

1. **Leaf-wise growth ranked children by their parent’s gain.** `SKPendingLeaf` stored the
   gain that produced the leaf, so both children tied and the priority queue degenerated to
   node-index order; a lower-gain child could be split before a higher-gain one.
   Fixed in `89b5b39` (`growth/leaf_wise.rs` now scans the candidates when the leaf is created
   and stores its own best candidate). Guard: `hist_growth_splits_the_leaf_with_the_largest_own_gain`.
2. **Multiclass boosting never refreshed its gradients.** `sk_run_histogram_multiclass`
   recomputed the derivatives from the initial class priors every stage, so every stage
   refitted identical trees; `subsample` was also silently ignored because growth received all
   training rows. Fixed in `be8faee`. Guards: `hist_multiclass_refreshes_gradients_between_stages`,
   `hist_multiclass_applies_row_subsampling`.
3. **Early stopping silently truncated to one stage without a validation split.** When the
   seeded hold-out was impossible the fit fell back to every row but still truncated to the
   monitor’s initial best count (`1`). Fixed in `27d2451` (monitor only exists when validation
   rows exist). Guard: `hist_early_stop_keeps_every_stage_when_no_validation_split_is_possible`.
4. **`SKBinnedDataset::build` accepted an out-of-range `maximum_bins`.** It clamped the binning
   internally but stored the raw value, so `0` panicked on a missing bin and `>255` left the
   missing bin outside the histogram stride. Fixed in `2d4b63e` (`InvalidHyperparameter`).
   Guard: `hist_dataset_rejects_out_of_range_maximum_bins`.
5. **Scenario tests tightened.** The leaf-budget assertion now demands exactly three leaves;
   unknown-category routing is asserted explicitly (`hist_growth_routes_an_unknown_category_away_from_the_left_mask`);
   formatted in `cce2734`.
6. **Design wording corrected.** `design.md` decision 5 claimed a missing-free node records
   “left”; the implementation deterministically records right (`default_left = false`), as the
   book already documented.

Companion documentation updated on `docs/histogram-gradient-boosting-book` (`deda3b5`) to
describe the own-gain priority, the refreshed multiclass gradients, per-stage subsampling,
the early-stopping fallback and the new tests.

## 5. Gates

```
cargo fmt --all --check                              -> clean
cargo clippy --workspace --all-targets -- -D warnings -> clean
cargo test -p sciencekit_ensemble                    -> 177 passed, 0 failed (+2 doc-tests)
cargo test --workspace                               -> every suite ok, 0 failed
```

One unrelated pre-existing flake was observed once:
`bagging::tree_logistic_and_neighbor_classifiers_compose_and_record_backend_span` (from
PR #318, untouched here) missed the backend span when prediction ran on rayon workers under
the thread-local subscriber. Three consecutive re-runs of the suite and the full workspace
run were green. Recorded so CI triage does not attribute it to this change.

## 6. Residual observations

- The `SKHistGradientBoosting` abbreviation and the two files over 200 lines are accepted as
  pre-existing/planning debt (§2).
- No histogram-vs-exact split tolerance is documented (§3); a future change can add a
  cross-splitter tolerance test if API users need that guarantee.
