//! Pure dispatcher for the halving grid search builders.

mod classifier_builder;
mod regressor_builder;
mod state;

pub use classifier_builder::SKHalvingGridSearchClassifierBuilder;
pub use regressor_builder::SKHalvingGridSearchRegressorBuilder;

#[cfg(test)]
mod halving_builder_tests;
