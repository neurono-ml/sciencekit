//! Explicit export refusal for the type-erased graph pipeline.
//!
//! A fitted graph stores its stages behind `Any` boxes (`SKBoxedFittedStage`),
//! so their concrete state types are not recoverable at export time. Returning
//! a typed error is the contract's required behaviour for a model that cannot
//! describe its learned state.

use sciencekit_common::{SKError, SKExportable, SKFloat, SKModelState};

use super::SKPipelineGraphModel;

impl<F: SKFloat> SKExportable<F> for SKPipelineGraphModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        Err(SKError::UnsupportedOperation {
            operation: "export_state",
            reason: "a graph pipeline stores its fitted stages behind type-erased boxes, so their \
                     concrete export state cannot be recovered; export each stage model before \
                     fitting the graph, or use the typed pipeline"
                .to_string(),
        })
    }
}
