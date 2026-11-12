//! Acceptance coverage for typed pipelines, feature unions and DAG pipelines
//! (specs `typed-pipeline`, `feature-union`, `pipeline-graph`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares its submodules.

mod feature_union_large_tests;
mod feature_union_small_tests;
mod fixtures;
mod large_data_tests;
mod pipeline_graph_large_tests;
mod pipeline_graph_small_tests;
mod pipeline_graph_validation_tests;
mod small_data_tests;
mod streaming_pipeline_tests;
mod transformer_coverage_tests;
