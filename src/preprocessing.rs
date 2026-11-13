//! The `preprocessing` family: scaling, encoding and feature expansion.
//!
//! Every item here resolves to `sciencekit_preprocessing`. The scaler modules
//! keep their internal split between `builder`, `core_implementation`, `model`
//! and `streaming`; the headline builder, estimator and model are also
//! re-exported directly so typical code does not need the inner path.

pub use sciencekit_preprocessing::minmax_scaler::{
    builder::SKMinMaxScalerBuilder, core_implementation::SKMinMaxScaler, model::SKMinMaxScalerModel,
};
pub use sciencekit_preprocessing::one_hot_encoder::{
    SKEncodedOutput, SKIntegerCategoryTable, SKOneHotColumnCategories, SKOneHotEncoder,
    SKOneHotEncoderBuilder, SKOneHotEncoderModel, SKUnknownPolicy,
};
pub use sciencekit_preprocessing::polynomial_features::{
    SKExpandedOutput, SKPolynomialFeatures, SKPolynomialFeaturesBuilder, SKPolynomialFeaturesModel,
};
pub use sciencekit_preprocessing::robust_scaler::{
    builder::SKRobustScalerBuilder, core_implementation::SKRobustScaler, model::SKRobustScalerModel,
};
pub use sciencekit_preprocessing::standard_scaler::{
    builder::SKStandardScalerBuilder, core_implementation::SKStandardScaler,
    model::SKStandardScalerModel,
};
pub use sciencekit_preprocessing::{
    minmax_scaler, one_hot_encoder, polynomial_features, robust_scaler, standard_scaler,
};
