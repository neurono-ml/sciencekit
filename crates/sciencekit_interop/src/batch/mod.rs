//! Column-major batches and their conversion into the library's views (spec
//! `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod categorical_extraction;
mod conversions;
mod feature_layout;
mod materialization;
mod tabular_batch;
mod target_extraction;
mod validation;

pub use categorical_extraction::SKCategoricalExtraction;
pub use feature_layout::SKFeatureLayout;
pub use tabular_batch::SKTabularBatch;
pub use target_extraction::SKTargetExtraction;
pub use validation::sk_validate_numeric_feature_columns;

#[cfg(test)]
mod categorical_extraction_tests;
#[cfg(test)]
mod conversions_tests;
#[cfg(test)]
mod materialization_tests;
#[cfg(test)]
mod tabular_batch_tests;
#[cfg(test)]
mod target_extraction_tests;
#[cfg(test)]
mod validation_tests;
