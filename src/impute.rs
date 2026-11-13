//! The `impute` family: missing-data transformers.
//!
//! Every item here resolves to `sciencekit_impute`. The KNN imputer is the
//! documented dense-only exception of its crate; the umbrella does not change
//! that policy, it only re-exports it.

pub use sciencekit_impute::knn::{
    SKImputationWeights, SKKNNImputer, SKKNNImputerBuilder, SKKNNImputerModel, sk_weighted_average,
};
pub use sciencekit_impute::missing_indicator::{
    SKIndicatorOutput, SKMissingFeatureScope, SKMissingIndicator, SKMissingIndicatorBuilder,
    SKMissingIndicatorModel,
};
pub use sciencekit_impute::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};
pub use sciencekit_impute::simple_imputer::{
    SKImputationStrategy, SKImputedOutput, SKSimpleImputer, SKSimpleImputerBuilder,
    SKSimpleImputerModel,
};
pub use sciencekit_impute::{knn, missing_indicator, missingness, simple_imputer};
