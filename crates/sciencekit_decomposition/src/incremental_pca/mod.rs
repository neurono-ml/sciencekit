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
mod merge;
mod sparse;

pub use builder::SKIncrementalPCABuilder;
pub use core_implementation::SKIncrementalPCA;

#[cfg(test)]
mod incremental_accumulator_tests;
#[cfg(test)]
mod incremental_builder_tests;
#[cfg(test)]
mod incremental_finalize_tests;
#[cfg(test)]
mod incremental_merge_tests;
#[cfg(test)]
mod incremental_sparse_tests;
