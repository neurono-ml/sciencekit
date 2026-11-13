//! Mapping an affine ONNX graph (`MatMul` + optional `Add` + a link) to a
//! linear or logistic state (spec `native-model-import`).
//!
//! Pure dispatcher: declares the mapping pieces and re-exports the entry point.

mod decision;
mod linear;
mod logistic;
mod shared;

use sciencekit_common::{SKFloat, SKModelState};

use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;
use crate::native::state::metadata::SKOnnxNativeMetadata;

/// Try to map the graph as an affine model; `None` when it is not one.
pub(crate) fn state_from_affine<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: Option<&SKOnnxNativeMetadata>,
) -> Result<Option<SKModelState<F>>, SKOnnxNativeError> {
    let Some(affine) = decision::read_affine(model)? else {
        return Ok(None);
    };
    let is_classifier = affine.link
        || metadata.is_some_and(|metadata| {
            metadata.estimator_name.contains("Classifier")
                || metadata.estimator_name.contains("Logistic")
        });
    if is_classifier {
        Ok(Some(logistic::state(metadata, affine)?))
    } else {
        Ok(Some(linear::state(affine)?))
    }
}
