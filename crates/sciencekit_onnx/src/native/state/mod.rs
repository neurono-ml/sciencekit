//! Pure dispatcher: map an ONNX graph to the neutral state a native model is
//! rebuilt from. Carries no implementation logic.

mod affine;
mod initializers;
mod metadata;
mod scalers;
mod trees;
mod validation;

use sciencekit_common::{SKFloat, SKModelState};

use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;

pub(crate) use metadata::SKOnnxNativeMetadata;

/// Map a decoded, validated ONNX graph to a neutral sciencekit state.
///
/// Scalers are recognized from their named initializers, tree ensembles from
/// the ONNX-ML node, and everything else as an affine decision function.
pub(crate) fn sk_state_from_onnx<F: SKFloat>(
    model: &SKOnnxModel,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let metadata = SKOnnxNativeMetadata::read(model);
    if let Some(metadata) = metadata.as_ref() {
        if metadata.format_version != 0
            && metadata.format_version != sciencekit_common::SK_MODEL_FORMAT_VERSION
        {
            return Err(SKOnnxNativeError::UnsupportedOperator {
                operator: "sciencekit.format_version".to_string(),
                reason: format!(
                    "the graph declares model format version {}, this build reads {}",
                    metadata.format_version,
                    sciencekit_common::SK_MODEL_FORMAT_VERSION
                ),
            });
        }
        if let Some(state) = scalers::state_from_scaler(model, metadata)? {
            return Ok(state);
        }
    }
    if let Some(state) = trees::state_from_trees(model, metadata.as_ref())? {
        return Ok(state);
    }
    if let Some(state) = affine::state_from_affine(model, metadata.as_ref())? {
        return Ok(state);
    }
    Err(SKOnnxNativeError::UnsupportedOperator {
        operator: metadata
            .map(|metadata| metadata.estimator_name)
            .unwrap_or_else(|| "graph".to_string()),
        reason: "the graph is not an affine model, a scaler or a tree ensemble".to_string(),
    })
}
