//! The estimator-name dispatcher from a neutral model state to an ONNX graph.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::SKOnnxModel;

/// Translate a neutral model state into an ONNX graph.
///
/// Dispatch happens exactly once, on the state's estimator name. A family with
/// no exporter returns `UnsupportedOperation` naming the family instead of a
/// partial graph.
pub fn sk_export_state_to_onnx<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<SKOnnxModel, SKError> {
    let estimator_name = state.metadata().estimator_name();
    Err(SKError::UnsupportedOperation {
        operation: "onnx_export",
        reason: format!(
            "the estimator family `{estimator_name}` has no ONNX exporter yet; exported graphs cover the families listed by `sk_onnx_coverage`"
        ),
    })
}
