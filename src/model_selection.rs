//! The `model_selection` family: splitting, cross-validation and search.
//!
//! Every item here resolves to `sciencekit_model_selection`.

pub use sciencekit_model_selection::cross_validation::{
    sk_cross_val_score_classifier, sk_cross_val_score_regressor,
};
pub use sciencekit_model_selection::gather::{SKTargetPartition, sk_gather_rows};
pub use sciencekit_model_selection::grid_search::{
    SKClassifierScoring, SKErrorScore, SKGridSearchCVClassifier, SKGridSearchCVClassifierBuilder,
    SKGridSearchCVClassifierModel, SKGridSearchCVRegressor, SKGridSearchCVRegressorBuilder,
    SKGridSearchCVRegressorModel, SKGridSearchResults, SKRegressorScoring, SKScoringFamily,
};
pub use sciencekit_model_selection::halving::{
    SKHalvingGridSearchClassifier, SKHalvingGridSearchClassifierBuilder,
    SKHalvingGridSearchClassifierModel, SKHalvingGridSearchRegressor,
    SKHalvingGridSearchRegressorBuilder, SKHalvingGridSearchRegressorModel, SKHalvingResource,
    SKHalvingRound, SKHalvingScoring, sk_halving_candidate_schedule, sk_halving_resource_schedule,
};
pub use sciencekit_model_selection::index_split::SKIndexSplit;
pub use sciencekit_model_selection::shuffle::sk_shuffle_indices;
pub use sciencekit_model_selection::split::SKSplitter;
pub use sciencekit_model_selection::split::group::{SKGroupKFold, SKGroupKFoldBuilder};
pub use sciencekit_model_selection::split::kfold::{SKKFold, SKKFoldBuilder};
pub use sciencekit_model_selection::split::stratified::{
    SKStratifiedKFold, SKStratifiedKFoldBuilder,
};
pub use sciencekit_model_selection::train_test_split::{
    SKPartitionSize, SKTrainTestSplit, SKTrainTestSplitBuilder, SKTrainTestSplitter,
    sk_train_test_split,
};
pub use sciencekit_model_selection::{
    cross_validation, gather, grid_search, halving, index_split, shuffle, split, train_test_split,
};
