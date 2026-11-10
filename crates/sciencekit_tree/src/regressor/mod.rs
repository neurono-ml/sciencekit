//! The CART regressor (spec `decision-trees`).

mod builder;
mod core_implementation;
mod criterion;
mod fit;
mod model;
mod predict;

pub use builder::SKDecisionTreeRegressorBuilder;
pub use core_implementation::SKDecisionTreeRegressor;
pub use model::SKDecisionTreeRegressorModel;

#[cfg(test)]
mod regressor_builder_tests;
#[cfg(test)]
mod regressor_tests;
