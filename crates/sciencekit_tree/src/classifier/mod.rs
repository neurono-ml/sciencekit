//! The CART classifier (spec `decision-trees`).

mod builder;
mod core_implementation;
mod criterion;
mod fit;
mod model;
mod predict;

pub use builder::SKDecisionTreeClassifierBuilder;
pub use core_implementation::SKDecisionTreeClassifier;
pub use model::SKDecisionTreeClassifierModel;

#[cfg(test)]
mod classifier_builder_tests;
#[cfg(test)]
mod classifier_fit_tests;
#[cfg(test)]
mod classifier_predict_tests;
#[cfg(test)]
mod classifier_weights_tests;
