//! Loading a model from bytes, and analysing its interface.

use std::path::Path;

use serde_onnx::ir::TensorStorage;

use super::SKOnnxImportError;
use super::interface::interface;
use super::interpreter::sk_onnx_supports_operator;
use crate::graph::SKOnnxModel;

impl SKOnnxModel {
    /// Decode and validate a model from its protobuf bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, SKOnnxImportError> {
        let bytes = bytes.into();
        let model = super::decode::decode_model(&bytes)?;
        let model = SKOnnxModel::from_parts(bytes, model)?;
        validate(&model)?;
        Ok(model)
    }

    /// Decode and validate a model from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SKOnnxImportError> {
        let bytes =
            std::fs::read(path).map_err(|error| SKOnnxImportError::Central(error.into()))?;
        SKOnnxModel::from_bytes(bytes)
    }
}

/// Validate every construct the importer cannot execute.
pub(crate) fn validate(model: &SKOnnxModel) -> Result<(), SKOnnxImportError> {
    let graph = &model.model().graph;
    if graph.nodes.is_empty() {
        return Err(SKOnnxImportError::UnsupportedOutput {
            detail: "the graph declares no operator nodes".to_string(),
        });
    }
    if graph.outputs.is_empty() {
        return Err(SKOnnxImportError::UnsupportedOutput {
            detail: "the graph declares no outputs".to_string(),
        });
    }
    for initializer in &graph.initializers {
        if matches!(initializer.data, TensorStorage::External(_)) {
            return Err(SKOnnxImportError::ExternalData {
                tensor: initializer.name.clone(),
            });
        }
    }
    for (index, node) in graph.nodes.iter().enumerate() {
        if !sk_onnx_supports_operator(&node.domain, &node.op_type) {
            return Err(SKOnnxImportError::UnsupportedOperator {
                domain: node.domain.clone(),
                operator: node.op_type.clone(),
                node_index: index,
            });
        }
    }
    for opset in &model.model().opset_import {
        if serde_onnx::proto::supported_opset(&opset.domain).is_none() {
            return Err(SKOnnxImportError::UnsupportedOpset {
                domain: opset.domain.clone(),
                declared: opset.version,
                supported: serde_onnx::proto::SUPPORTED_ONNX_OPSET,
            });
        }
    }
    interface(model)?;
    Ok(())
}
