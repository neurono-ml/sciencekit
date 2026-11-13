//! The serialized ONNX model and its declared interface.

use std::collections::BTreeMap;
use std::path::Path;

use sciencekit_common::SKError;

use super::SKOnnxValueInfo;

/// A serialized ONNX model.
///
/// The bytes are a complete `ModelProto`; the declared inputs and outputs are
/// kept beside them so a caller can inspect the interface without decoding.
#[derive(Debug, Clone)]
pub struct SKOnnxModel {
    bytes: Vec<u8>,
    opset: i64,
    onnx_ml_opset: i64,
    inputs: Vec<SKOnnxValueInfo>,
    outputs: Vec<SKOnnxValueInfo>,
    metadata: BTreeMap<String, String>,
    node_count: usize,
    initializer_count: usize,
}

impl SKOnnxModel {
    /// Assemble a model from a builder's validated parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        bytes: Vec<u8>,
        opset: i64,
        onnx_ml_opset: i64,
        inputs: Vec<SKOnnxValueInfo>,
        outputs: Vec<SKOnnxValueInfo>,
        metadata: BTreeMap<String, String>,
        node_count: usize,
        initializer_count: usize,
    ) -> Self {
        SKOnnxModel {
            bytes,
            opset,
            onnx_ml_opset,
            inputs,
            outputs,
            metadata,
            node_count,
            initializer_count,
        }
    }

    /// The serialized `ModelProto` bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the model and return its bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// The declared default-domain opset version.
    pub fn opset(&self) -> i64 {
        self.opset
    }

    /// The declared ONNX-ML opset version.
    pub fn onnx_ml_opset(&self) -> i64 {
        self.onnx_ml_opset
    }

    /// The declared graph inputs.
    pub fn inputs(&self) -> &[SKOnnxValueInfo] {
        &self.inputs
    }

    /// The declared graph outputs.
    pub fn outputs(&self) -> &[SKOnnxValueInfo] {
        &self.outputs
    }

    /// The model metadata entries.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// The number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The number of graph initializers.
    pub fn initializer_count(&self) -> usize {
        self.initializer_count
    }

    /// Write the serialized bytes to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SKError> {
        std::fs::write(path, &self.bytes).map_err(SKError::Io)
    }
}
