//! Runtime directed acyclic graph pipelines over [`crate::SKFeatureBlock`]
//! (spec `pipeline-graph`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod builder_accessors;
mod final_estimator;
mod graph_definition;
mod node;

pub use builder::SKPipelineGraphBuilder;
pub use final_estimator::SKBoxedFinalModel;
pub use graph_definition::SKPipelineGraph;
pub use node::{SKNodeIdentifier, SKPipelineGraphNodeKind};

#[cfg(test)]
mod graph_builder_tests;
