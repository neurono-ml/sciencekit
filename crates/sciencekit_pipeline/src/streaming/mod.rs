//! Streaming pipelines over a reopenable feature source (spec
//! `streaming-pipeline`).
//!
//! A streaming pipeline composes [`SKStreamingTransformer`] stages that are
//! fitted by reading owned batches from an [`SKSourceProvider`] and then apply
//! one batch at a time, so the whole dataset is never resident. The pipeline
//! reuses the shared prefetch executor for its I/O and keeps every stage's
//! learned state in its fitted model.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod errors;
mod finalizer;
mod fitting;
mod memory;
mod pipeline;
mod provider;
mod stage;
mod transformer;

pub use builder::{SKStreamingPipeline, SKStreamingPipelineBuilder};
pub use pipeline::{
    SKFittedStreamingPipeline, SKStreamingFitReport, SKStreamingPipelineConfigured,
};
pub use transformer::SKStreamingTransformer;

#[cfg(test)]
mod builder_tests;

#[cfg(test)]
mod epochs_tests;

#[cfg(test)]
mod fit_tests;

#[cfg(test)]
mod oneshot_tests;

#[cfg(test)]
pub(crate) mod testing;

#[cfg(test)]
mod transform_tests;

#[cfg(test)]
mod transformer_tests;
