//! Incremental Principal Component Analysis (`SKIncrementalPCA`, spec
//! `incremental-pca`).
//!
//! Pure dispatcher module: submodules hold configuration, Chan-Golub-LeVeque
//! scatter accumulation, sparse Gram updates, symmetric-eigensolver
//! finalization, builder, fitting, streaming and projection.

mod accumulator;
mod builder;
mod config;
mod core_implementation;
mod finalize;
mod fit;
mod merge;
mod model;
mod partial_fit;
mod sparse;
mod streaming;
mod transform;
mod validation;

pub use builder::SKIncrementalPCABuilder;
pub use core_implementation::SKIncrementalPCA;
pub use model::SKIncrementalPCAModel;

#[cfg(test)]
mod incremental_accumulator_tests;
#[cfg(test)]
mod incremental_builder_tests;
#[cfg(test)]
mod incremental_finalize_tests;
#[cfg(test)]
mod incremental_fit_tests;
#[cfg(test)]
mod incremental_merge_tests;
#[cfg(test)]
mod incremental_sparse_tests;
#[cfg(test)]
mod incremental_streaming_tests;
#[cfg(test)]
mod incremental_transform_tests;
