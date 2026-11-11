//! `SKFeatureUnion` — parallel feature branches concatenated column-wise
//! (spec `feature-union`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file.

mod accessors;
mod append;
mod boxed;
mod builder;
mod concatenate;
mod errors;
mod model;
mod pipeline_stage;
mod stages;

pub use append::SKAppendUnionStage;
pub use boxed::{SKBlockTransformer, SKBoxedFittedStage};
pub use builder::{SKFeatureUnion, SKFeatureUnionBuilder};
pub use errors::SKFeatureUnionMemberError;
pub use model::SKFeatureUnionModel;
pub use stages::{SKFeatureUnionArity, SKFeatureUnionStages};

#[cfg(test)]
mod union_boxed_tests;
#[cfg(test)]
mod union_concatenation_tests;
#[cfg(test)]
mod union_fit_tests;
#[cfg(test)]
mod union_model_tests;
#[cfg(test)]
mod union_pipeline_tests;
#[cfg(test)]
mod union_stages_tests;
#[cfg(test)]
mod union_validation_fixtures_tests;
#[cfg(test)]
mod union_validation_tests;
