//! `sciencekit_pipeline` — compile-time typed linear pipelines (spec
//! `typed-pipeline`).
//!
//! A pipeline is composed through a builder whose **type** records the composed
//! stages, so an incompatible chain does not compile. This first layer carries
//! the pipeline medium: the owned dense/sparse [`SKFeatureBlock`] that flows
//! between stages without losing its representation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block;

pub use block::{SKFeatureBlock, SKIntoFeatureBlock};

#[cfg(test)]
mod block_tests;
