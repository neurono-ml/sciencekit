//! `SKSimpleImputer`: univariate missing-value imputation (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod builder;
mod core_implementation;
mod model;
mod output;
mod statistics;
mod strategy;

pub use builder::SKSimpleImputerBuilder;
pub use core_implementation::SKSimpleImputer;
pub use model::SKSimpleImputerModel;
pub use output::SKImputedOutput;
pub use strategy::SKImputationStrategy;

#[cfg(test)]
mod builder_tests;

#[cfg(test)]
mod fit_tests;

#[cfg(test)]
mod output_tests;

#[cfg(test)]
mod partial_fit_tests;

#[cfg(test)]
mod transform_tests;
