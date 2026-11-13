//! The serialized ONNX model and its declared interface.

use std::collections::BTreeMap;
use std::path::Path;

use sciencekit_common::SKError;
use serde_onnx::ir::Model;

use super::SKOnnxValueInfo;

/// A serialized ONNX model.
///
/// Both directions share this handle: the exporter builds it from a validated
/// graph, the importer decodes it from bytes. The decoded
/// [`serde_onnx::ir::Model`] stays beside the bytes so callers can inspect the
/// interface without decoding again.
#[derive(Debug, Clone)]
pub struct SKOnnxModel {
    bytes: Vec<u8>,
    model: Model,
    inputs: Vec<SKOnnxValueInfo>,
    outputs: Vec<SKOnnxValueInfo>,
}

impl SKOnnxModel {
    /// Assemble a model from encoded bytes and its decoded shared IR.
    pub(crate) fn from_parts(bytes: Vec<u8>, model: Model) -> Result<Self, SKError> {
        let inputs = model
            .graph
            .inputs
            .iter()
            .map(SKOnnxValueInfo::from_ir)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = model
            .graph
            .outputs
            .iter()
            .map(SKOnnxValueInfo::from_ir)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SKOnnxModel {
            bytes,
            model,
            inputs,
            outputs,
        })
    }

    /// Assemble a model without running the import validation (tests only).
    #[cfg(test)]
    pub(crate) fn from_parts_in_tests(bytes: Vec<u8>, model: Model) -> Self {
        SKOnnxModel::from_parts(bytes, model).expect("the fixture interface converts")
    }

    /// The shared ONNX IR this model was decoded into.
    pub fn model(&self) -> &Model {
        &self.model
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
        self.model.opset("").unwrap_or(super::SK_ONNX_DEFAULT_OPSET)
    }

    /// The declared ONNX-ML opset version.
    pub fn onnx_ml_opset(&self) -> i64 {
        self.model
            .opset(super::SK_ONNX_ML_DOMAIN)
            .unwrap_or(super::SK_ONNX_ML_TREE_OPSET)
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
    pub fn metadata(&self) -> BTreeMap<String, String> {
        self.model.metadata.iter().cloned().collect()
    }

    /// The number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.model.graph.nodes.len()
    }

    /// The number of graph initializers.
    pub fn initializer_count(&self) -> usize {
        self.model.graph.initializers.len()
    }

    /// Write the serialized bytes to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SKError> {
        std::fs::write(path, &self.bytes).map_err(SKError::Io)
    }
}
