//! The missing-indicator estimator and its mask (spec `missing-indicator`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod builder;
mod core_implementation;
mod model;
mod output;
mod scope;

pub use builder::SKMissingIndicatorBuilder;
pub use core_implementation::SKMissingIndicator;
pub use model::SKMissingIndicatorModel;
pub use output::SKIndicatorOutput;
pub use scope::SKMissingFeatureScope;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod builder_tests;

#[cfg(test)]
mod fit_tests;

#[cfg(test)]
mod transform_tests;
