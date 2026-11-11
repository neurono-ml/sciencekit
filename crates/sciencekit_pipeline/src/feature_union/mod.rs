//! `SKFeatureUnion` — parallel feature branches concatenated column-wise
//! (spec `feature-union`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file.

mod append;
mod builder;
mod concatenate;
mod model;
mod stages;

pub use append::SKAppendUnionStage;
pub use builder::{SKFeatureUnion, SKFeatureUnionBuilder};
pub use model::SKFeatureUnionModel;
pub use stages::{SKFeatureUnionArity, SKFeatureUnionStages};

#[cfg(test)]
mod union_concat_tests;
#[cfg(test)]
mod union_fit_tests;
#[cfg(test)]
mod union_stages_tests;
