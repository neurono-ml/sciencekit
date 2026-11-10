//! `SKKNNImputer`: nearest-neighbor missing-value imputation
//! (spec `nearest-neighbor-imputation`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod builder;
mod core_implementation;
mod distance_block;
mod donors;
mod fit;
mod marker;
mod model;
mod validation;
mod weights;

pub use builder::SKKNNImputerBuilder;
pub use core_implementation::SKKNNImputer;
pub use model::SKKNNImputerModel;
pub use weights::{SKImputationWeights, sk_weighted_average};

#[cfg(test)]
mod builder_tests;

#[cfg(test)]
mod distance_block_tests;

#[cfg(test)]
mod fit_tests;

#[cfg(test)]
mod fit_span_tests;

#[cfg(test)]
mod marker_tests;
