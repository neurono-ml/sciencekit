//! Column-major batches and their conversion into the library's views (spec
//! `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod tabular_batch;
mod validation;

pub use tabular_batch::SKTabularBatch;
pub use validation::sk_validate_numeric_feature_columns;

#[cfg(test)]
mod tabular_batch_tests;
#[cfg(test)]
mod validation_tests;
