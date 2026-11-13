//! The CART regressor (spec `decision-trees`).

mod builder;
mod core_implementation;
mod criterion;
mod export;
mod fit;
mod model;
mod predict;

pub use builder::SKDecisionTreeRegressorBuilder;
pub use core_implementation::SKDecisionTreeRegressor;
pub use export::SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME;
pub use model::SKDecisionTreeRegressorModel;

#[cfg(test)]
mod regressor_builder_tests;
#[cfg(test)]
mod regressor_tests;
