//! The CART classifier (spec `decision-trees`).

mod builder;
mod core_implementation;
mod criterion;
mod export;
mod fit;
mod model;
mod predict;
pub(crate) mod validation;

pub use builder::SKDecisionTreeClassifierBuilder;
pub use core_implementation::SKDecisionTreeClassifier;
pub use export::SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME;
pub use model::SKDecisionTreeClassifierModel;

#[cfg(test)]
mod classifier_builder_tests;
#[cfg(test)]
mod classifier_fit_tests;
#[cfg(test)]
mod classifier_predict_tests;
#[cfg(test)]
mod classifier_weights_tests;

#[cfg(test)]
mod export_tests;
