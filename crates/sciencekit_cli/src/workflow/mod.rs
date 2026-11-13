//! The versioned workflow format and its runner.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

pub(crate) mod error;
pub(crate) mod loader;
pub(crate) mod runner;
pub(crate) mod schema;
pub(crate) mod steps;
pub(crate) mod validation;

#[allow(unused_imports)]
pub use error::SKWorkflowError;
#[allow(unused_imports)]
pub use loader::{SKWorkflowFormat, sk_load_workflow, sk_parse_workflow};
#[allow(unused_imports)]
pub use runner::sk_execute_workflow;
#[allow(unused_imports)]
pub use schema::{
    SK_WORKFLOW_SUPPORTED_VERSION, SK_WORKFLOW_VERSION_PATH, SKWorkflow, SKWorkflowDataset,
    SKWorkflowEstimator, SKWorkflowOutput, SKWorkflowPipelineStep, SKWorkflowSearch,
};
#[allow(unused_imports)]
pub use steps::{SKGridSearchOutcome, sk_apply_pipeline_steps, sk_grid_search};
#[allow(unused_imports)]
pub use validation::{sk_validate_workflow_semantics, sk_validate_workflow_value};

#[cfg(test)]
mod schema_tests;
