//! Shared fixtures for the streaming pipeline tests.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! the test transformers, estimators and sources; every implementation lives in
//! its own file.

mod test_column_scaling;
mod test_estimators;
mod test_execution_plan;
mod test_mean_centering;
mod test_sources;

pub(crate) use test_column_scaling::SKTestColumnScaling;
pub(crate) use test_estimators::{
    SKTestIdentityEstimator, SKTestIdentityModel, SKTestIdentityOutput,
};
pub(crate) use test_execution_plan::sk_test_execution_plan;
pub(crate) use test_mean_centering::SKTestMeanCentering;
pub(crate) use test_sources::{
    SKTestBlocksSource, sk_reopenable_blocks, sk_single_block, sk_tiny_blocks,
};
