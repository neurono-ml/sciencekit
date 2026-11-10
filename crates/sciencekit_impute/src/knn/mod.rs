//! `SKKNNImputer`: nearest-neighbor missing-value imputation
//! (spec `nearest-neighbor-imputation`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod builder;
mod core_implementation;
mod weights;

pub use builder::SKKNNImputerBuilder;
pub use core_implementation::SKKNNImputer;
pub use weights::{SKImputationWeights, sk_weighted_average};

#[cfg(test)]
mod builder_tests;
