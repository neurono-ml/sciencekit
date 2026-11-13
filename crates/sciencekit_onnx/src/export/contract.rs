//! The `SKToOnnx` contract (spec `onnx-export`, requirement "ONNX export
//! contract").

use sciencekit_common::{SKError, SKExportable, SKFloat};

use crate::graph::SKOnnxModel;

use super::sk_export_state_to_onnx;

/// A fitted model that can produce an ONNX graph.
///
/// The contract is deliberately thin: a model already describes itself through
/// [`SKExportable`], so ONNX export is a translation of that neutral state, not a
/// second model format. The blanket implementation below gives every exportable
/// model the method without touching its own crate.
pub trait SKToOnnx<F: SKFloat> {
    /// Translate the fitted model into a serialized ONNX graph.
    fn to_onnx(&self) -> Result<SKOnnxModel, SKError>;
}

impl<F, T> SKToOnnx<F> for T
where
    F: SKFloat,
    T: SKExportable<F>,
{
    fn to_onnx(&self) -> Result<SKOnnxModel, SKError> {
        let state = self.export_state()?;
        sk_export_state_to_onnx(&state)
    }
}
