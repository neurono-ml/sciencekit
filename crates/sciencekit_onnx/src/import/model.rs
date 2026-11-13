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
    structural(model)?;
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

/// Run the shared IR's structural validator and translate its diagnoses.
///
/// A cycle is the construct the interpreter cannot explain on its own: walking
/// a graph in dependency order would stall and report the first unproduced
/// operand, hiding the real defect. Duplicated outputs and malformed shapes are
/// likewise rejected before inference.
fn structural(model: &SKOnnxModel) -> Result<(), SKOnnxImportError> {
    use serde_onnx::ir::ValidationError;
    let report = serde_onnx::ir::validate_model(model.model());
    // A cycle also surfaces as an undefined value in the validator's linear
    // scan, so the cycle diagnosis takes precedence over the symptom.
    if let Some(ValidationError::Cycle { path }) = report
        .errors
        .iter()
        .find(|error| matches!(error, ValidationError::Cycle { .. }))
    {
        return Err(SKOnnxImportError::GraphCycle { path: path.clone() });
    }
    match report.errors.into_iter().next() {
        Some(ValidationError::UndefinedValue { node_index, name }) => {
            Err(SKOnnxImportError::MissingInitializer { name, node_index })
        }
        Some(other) => Err(SKOnnxImportError::InvalidModel {
            detail: other.to_string(),
        }),
        None => Ok(()),
    }
}
