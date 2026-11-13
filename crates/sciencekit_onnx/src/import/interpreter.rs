//! The documented interpreter: a deterministic executor over the shared IR.
//!
//! The interpreter is deliberately strict. It walks the graph in dependency
//! order, resolves every operand to an initializer or an earlier node output
//! and rejects any operator outside [`SK_ONNX_SUPPORTED_OPERATORS`].

use std::collections::BTreeMap;

use serde_onnx::ir::{Node, Tensor};

use super::SKOnnxImportError;
use super::dispatch::run_node;
use super::initializers::tensor_value;
use super::value::SKOnnxValue;

/// The default-domain operator set this importer executes.
pub const SK_ONNX_SUPPORTED_OPERATORS: &[&str] = &[
    "Abs",
    "Add",
    "ArgMax",
    "ArgMin",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "Div",
    "Equal",
    "Exp",
    "Flatten",
    "Gather",
    "Greater",
    "Identity",
    "Less",
    "MatMul",
    "Mul",
    "Neg",
    "Pow",
    "ReduceMean",
    "ReduceSum",
    "Reshape",
    "Sigmoid",
    "Softmax",
    "Squeeze",
    "Sub",
    "Tanh",
    "TopK",
    "Transpose",
    "Unsqueeze",
];

/// The ONNX-ML operators this importer executes.
pub const SK_ONNX_SUPPORTED_ML_OPERATORS: &[&str] =
    &["TreeEnsembleClassifier", "TreeEnsembleRegressor"];

/// Whether a node belongs to the documented subset.
pub fn sk_onnx_supports_operator(domain: &str, operator: &str) -> bool {
    match domain {
        "" | "ai.onnx" => SK_ONNX_SUPPORTED_OPERATORS.contains(&operator),
        "ai.onnx.ml" => SK_ONNX_SUPPORTED_ML_OPERATORS.contains(&operator),
        _ => false,
    }
}

/// Execute a graph and return every declared output by name.
pub(crate) fn execute(
    initializers: &[Tensor],
    nodes: &[Node],
    inputs: &BTreeMap<String, SKOnnxValue>,
    outputs: &[String],
) -> Result<BTreeMap<String, SKOnnxValue>, SKOnnxImportError> {
    let mut values = inputs.clone();
    for initializer in initializers {
        values.insert(initializer.name.clone(), tensor_value(initializer)?);
    }
    let mut pending: Vec<(usize, &Node)> = nodes.iter().enumerate().collect();
    while !pending.is_empty() {
        let position = pending.iter().position(|(_, node)| {
            node.inputs
                .iter()
                .all(|input| input.is_empty() || values.contains_key(input))
        });
        let Some(position) = position else {
            let (index, node) = pending[0];
            let missing = node
                .inputs
                .iter()
                .find(|input| !input.is_empty() && !values.contains_key(*input))
                .cloned()
                .unwrap_or_default();
            return Err(SKOnnxImportError::MissingInitializer {
                name: missing,
                node_index: index,
            });
        };
        let (index, node) = pending.remove(position);
        let produced = run_node(index, node, &values)?;
        for (name, value) in node.outputs.iter().zip(produced) {
            values.insert(name.clone(), value);
        }
    }
    outputs
        .iter()
        .map(|name| {
            values
                .get(name)
                .cloned()
                .map(|value| (name.clone(), value))
                .ok_or_else(|| SKOnnxImportError::MissingInitializer {
                    name: name.clone(),
                    node_index: nodes.len(),
                })
        })
        .collect()
}
