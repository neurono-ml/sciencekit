//! The CART classifier (spec `decision-trees`).

mod builder;
mod core_implementation;
mod fit;
mod model;
mod predict;

#[cfg(test)]
mod classifier_builder_tests;
#[cfg(test)]
mod classifier_fit_tests;
#[cfg(test)]
mod classifier_predict_tests;
#[cfg(test)]
mod classifier_weights_tests;
