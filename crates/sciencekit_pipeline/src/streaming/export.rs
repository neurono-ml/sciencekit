//! Explicit export refusal for the type-erased streaming pipeline.
//!
//! A fitted streaming pipeline keeps its stages in a `Vec<Box<dyn SKFittedStage>>`,
//! so their concrete state types are not recoverable at export time. Returning
//! a typed error is the contract's required behaviour for a model that cannot
//! describe its learned state.

use sciencekit_common::{SKError, SKExportable, SKFloat, SKModelState};

use super::SKFittedStreamingPipeline;

impl<F: SKFloat, FinalModel> SKExportable<F> for SKFittedStreamingPipeline<F, FinalModel> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        Err(SKError::UnsupportedOperation {
            operation: "export_state",
            reason: "a streaming pipeline stores its fitted stages behind trait objects, so their \
                     concrete export state cannot be recovered; re-fit with a typed pipeline when \
                     a portable state is required"
                .to_string(),
        })
    }
}
