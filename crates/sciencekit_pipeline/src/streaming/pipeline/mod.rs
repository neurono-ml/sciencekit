//! The configured and fitted streaming pipelines (spec `streaming-pipeline`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod configured;
mod fitted;
mod report;

pub use configured::SKStreamingPipelineConfigured;
pub use fitted::SKFittedStreamingPipeline;
pub use report::SKStreamingFitReport;
