//! Runtime directed acyclic graph pipelines over [`crate::SKFeatureBlock`]
//! (spec `pipeline-graph`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod builder_accessors;
mod classifier_wrapper;
mod errors;
mod execute;
mod final_estimator;
mod graph_definition;
mod merge;
mod model;
mod node;
mod order;
mod predict;
mod regressor_wrapper;
mod validate;

pub use builder::SKPipelineGraphBuilder;
pub use classifier_wrapper::SKPipelineGraphClassifier;
pub use errors::SKPipelineGraphNodeError;
pub use final_estimator::SKBoxedFinalModel;
pub use graph_definition::SKPipelineGraph;
pub use model::SKPipelineGraphModel;
pub use node::{SKNodeIdentifier, SKPipelineGraphNodeKind};
pub use order::SKNodeBlockStore;
pub use regressor_wrapper::SKPipelineGraphRegressor;

#[cfg(test)]
mod graph_builder_tests;
#[cfg(test)]
mod graph_concurrency_tests;
#[cfg(test)]
mod graph_fit_tests;
#[cfg(test)]
mod graph_merge_tests;
#[cfg(test)]
mod graph_order_tests;
#[cfg(test)]
mod graph_predict_tests;
#[cfg(test)]
mod graph_representation_tests;
#[cfg(test)]
mod graph_validate_tests;
