//! The `ensemble` family: forests, bagging and gradient boosting.
//!
//! Every item here resolves to `sciencekit_ensemble`. The split vocabulary
//! (`SKClassWeight`, `SKMaximumFeatures`, `SKTreeCriterion`) is owned by
//! `sciencekit_tree`; this crate re-exports it and so does the umbrella, so the
//! names in `ensemble` and `tree` are the same types, not competing ones.

pub use sciencekit_ensemble::bagging::{
    SKBaggingClassifier, SKBaggingClassifierBuilder, SKBaggingClassifierModel, SKBaggingError,
    SKBaggingRegressor, SKBaggingRegressorBuilder, SKBaggingRegressorModel, SKBaseEstimatorError,
};
pub use sciencekit_ensemble::gradient_boosting::{
    SKGradientBoostingClassificationLoss, SKGradientBoostingClassifier,
    SKGradientBoostingClassifierBuilder, SKGradientBoostingClassifierModel,
    SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
    SKGradientBoostingRegressorBuilder, SKGradientBoostingRegressorModel,
};
pub use sciencekit_ensemble::histogram_boosting::{
    SKHistGradientBoostingClassifier, SKHistGradientBoostingClassifierBuilder,
    SKHistGradientBoostingClassifierModel, SKHistGradientBoostingRegressionLoss,
    SKHistGradientBoostingRegressor, SKHistGradientBoostingRegressorBuilder,
    SKHistGradientBoostingRegressorModel,
};
pub use sciencekit_ensemble::random_forest::{
    SKRandomForestClassifier, SKRandomForestClassifierBuilder, SKRandomForestClassifierModel,
    SKRandomForestRegressor, SKRandomForestRegressorBuilder, SKRandomForestRegressorModel,
};
pub use sciencekit_ensemble::{
    SKClassWeight, SKMaximumFeatures, SKMaximumSamples, SKTreeCriterion,
};
pub use sciencekit_ensemble::{
    bagging, boosting, gradient_boosting, histogram_boosting, random_forest,
};
