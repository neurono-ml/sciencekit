//! `sciencekit_model_selection` — reproducible data partitioning for sciencekit
//! (spec `model-selection-splits`).
//!
//! The crate ships index-based train/test splitting, K-fold, stratified K-fold
//! and group K-fold. Splitters return [`SKIndexSplit`] values that address the
//! original rows, so no data is copied until a fold is materialized. Seeded
//! shuffling goes through the shared deterministic generator, so the same seed
//! and row count always produce the same partition.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cross_validation;
pub mod gather;
pub mod grid_search;
pub mod halving;
pub mod index_split;
pub mod shuffle;
pub mod split;
pub mod train_test_split;

pub use cross_validation::{sk_cross_val_score_classifier, sk_cross_val_score_regressor};
pub use gather::{SKTargetPartition, sk_gather_rows};
pub use grid_search::{
    SKClassifierScoring, SKErrorScore, SKGridSearchCVClassifier, SKGridSearchCVClassifierBuilder,
    SKGridSearchCVClassifierModel, SKGridSearchCVRegressor, SKGridSearchCVRegressorBuilder,
    SKGridSearchCVRegressorModel, SKGridSearchResults, SKRegressorScoring, SKScoringFamily,
};
pub use halving::{
    SKHalvingResource, SKHalvingRound, sk_halving_candidate_schedule, sk_halving_resource_schedule,
};
pub use index_split::SKIndexSplit;
pub use shuffle::sk_shuffle_indices;
pub use split::SKSplitter;
pub use split::group::{SKGroupKFold, SKGroupKFoldBuilder};
pub use split::kfold::{SKKFold, SKKFoldBuilder};
pub use split::stratified::{SKStratifiedKFold, SKStratifiedKFoldBuilder};
pub use train_test_split::{
    SKPartitionSize, SKTrainTestSplit, SKTrainTestSplitBuilder, SKTrainTestSplitter,
    sk_train_test_split,
};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod index_split_tests;
#[cfg(test)]
mod shuffle_tests;
#[cfg(test)]
mod validation_tests;
