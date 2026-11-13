# Proposal

## Why

Every later model-selection and evaluation workflow needs reproducible data partitioning.
The library has no train/test split, no K-fold iterator and no cross-validation helper, so
the only way to evaluate a model today is to partition by hand. This change supplies the
shared index-based split foundation that `SKGridSearchCV` and the pipeline examples build
on.

## What Changes

- New sub-crate `sciencekit_model_selection`.
- `sk_train_test_split`: split features (and optional targets) into train/test partitions
  with `test_size`/`train_size`, `shuffle`, `random_seed` and optional stratification.
- `SKKFold`, `SKStratifiedKFold` and `SKGroupKFold`, all builder-only, producing
  deterministic `SKIndexSplit` values (train/test index vectors).
- `SKSplitter<F>` contract used by grid search; the splitter receives optional targets and
  groups and returns index splits.
- `sk_cross_val_score` for classifier and regressor pipelines, fitting one model per fold
  and returning the per-fold scores.
- `sk_gather_rows` helper for dense and sparse feature blocks and targets, used by the
  splitters and grid search; no densification.
- Validation: invalid fractions, empty input, too many splits, class/group count smaller
  than the split count, length mismatches and non-finite values.
- Explicit non-goals: time-series splits, repeated K-fold, leave-one-out, nested CV,
  `memmap2` backing and export (Wave 7).
- Companion tests: hand-checkable partitions, deterministic shuffles, stratification
  proportions, group separation, density parity, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `model-selection-splits`: train/test splitting, K-fold, stratified K-fold, group K-fold,
  the index-split contract and the row-gathering helper.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_model_selection` workspace member; root `Cargo.toml`
  membership.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  `sciencekit_metrics` as a dev-dependency.
- **Downstream:** unblocks `SKGridSearchCV` (W6.4), `SKHalvingGridSearchCV` (W6.5) and
  cross-validated pipeline examples.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent split
  generation, metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export
  applies to fitted models, not to splitters, and is deferred to Wave 7.
