//! The CART regressor (spec `decision-trees`).

mod builder;
mod core_implementation;
mod fit;
mod model;
mod predict;

#[cfg(test)]
mod regressor_builder_tests;
#[cfg(test)]
mod regressor_tests;
