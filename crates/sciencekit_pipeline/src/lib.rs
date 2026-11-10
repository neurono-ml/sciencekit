//! `sciencekit_pipeline` — compile-time typed linear pipelines (spec
//! `typed-pipeline`).
//!
//! A pipeline is composed through a builder whose **type** records the composed
//! stages: [`SKPipeline::start`] opens a [`SKPipelineBuilder`], each `then`
//! appends a transformer stage (the scalar and the block medium are checked at
//! compile time), and a classifier or regressor finalizer accepts the supervised
//! estimator that consumes the last stage's output. Fitting walks the whole
//! chain once, and the fitted model implements the shared classifier / regressor
//! prediction contracts.
//!
//! Dense and sparse data flow as [`SKFeatureBlock`] values: the representation a
//! stage produces is the representation the next stage receives, and the
//! pipeline itself never densifies a sparse block.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block;
pub mod pipeline;
mod stage;
mod stages;

pub use block::{SKFeatureBlock, SKIntoFeatureBlock};
pub use pipeline::{
    SKPipeline, SKPipelineBuilder, SKPipelineClassifier, SKPipelineClassifierBuilder,
    SKPipelineRegressor, SKPipelineRegressorBuilder,
};
pub use stage::SKPipelineTransformer;
pub use stages::SKConfiguredStages;

#[cfg(test)]
mod block_tests;
#[cfg(test)]
mod stage_tests;
