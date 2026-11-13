//! The affine decision function recovered from a graph.

use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;

use super::super::initializers::{SKOnnxConstant, constant};
use super::super::validation::{sk_element_type_mismatch, sk_require_floating, sk_shape_mismatch};
use super::shared::invalid;

/// Operators an affine graph may use beside the mapped ones.
const AFFINE_OPERATORS: &[&str] = &[
    "MatMul", "Add", "Sigmoid", "Softmax", "ArgMax", "Sub", "Concat", "Identity", "Cast",
];

/// The `X · W + b` decision function plus whether a link operator follows it.
pub(crate) struct Affine {
    pub(crate) weights: SKOnnxConstant,
    pub(crate) bias: Option<SKOnnxConstant>,
    pub(crate) feature_count: usize,
    pub(crate) link: bool,
}

/// Read the affine structure, or `None` when the graph carries no `MatMul`.
pub(crate) fn read_affine(model: &SKOnnxModel) -> Result<Option<Affine>, SKOnnxNativeError> {
    let graph = &model.model().graph;
    for node in &graph.nodes {
        let mapped = AFFINE_OPERATORS.contains(&node.op_type.as_str())
            || node.op_type.starts_with("TreeEnsemble");
        if !mapped {
            return Err(SKOnnxNativeError::UnsupportedOperator {
                operator: node.op_type.clone(),
                reason: "the affine mapper only understands MatMul, Add and a link operator"
                    .to_string(),
            });
        }
    }
    let Some(matmul) = graph.nodes.iter().find(|node| node.op_type == "MatMul") else {
        return Ok(None);
    };
    let weight_name = matmul
        .inputs
        .get(1)
        .ok_or_else(|| invalid("the MatMul node has no weight operand"))?;
    let weights = constant(model, weight_name).ok_or_else(|| {
        invalid(format!(
            "the MatMul weight `{weight_name}` is not an initializer"
        ))
    })?;
    sk_require_floating(&weights, weight_name)?;
    let &[feature_count, heads] = weights.shape.as_slice() else {
        return Err(invalid(format!(
            "the MatMul weight must be a matrix, got shape {:?}",
            weights.shape
        )));
    };
    if heads == 0 || weights.values.len() != feature_count * heads {
        return Err(invalid(
            "the MatMul weight payload disagrees with its shape",
        ));
    }
    sk_validate_feature_input(model, &matmul.inputs[0], feature_count, &weights)?;
    let bias = find_bias(model, &matmul.outputs);
    if let Some(bias) = bias.as_ref() {
        sk_require_floating(bias, "MatMul.bias")?;
    }
    let link = graph
        .nodes
        .iter()
        .any(|node| matches!(node.op_type.as_str(), "Sigmoid" | "Softmax"));
    Ok(Some(Affine {
        weights,
        bias,
        feature_count,
        link,
    }))
}

fn sk_validate_feature_input(
    model: &SKOnnxModel,
    input_name: &str,
    feature_count: usize,
    weights: &SKOnnxConstant,
) -> Result<(), SKOnnxNativeError> {
    let Some(input) = model
        .inputs()
        .iter()
        .find(|input| input.name() == input_name)
    else {
        return Ok(());
    };
    let weight_type = weights
        .data_type
        .unwrap_or(crate::graph::SKOnnxTensorDType::Double);
    if input.element_type() != weight_type {
        return Err(sk_element_type_mismatch(
            input.name(),
            weight_type,
            input.element_type(),
        ));
    }
    if let Some(declared) = input.feature_count() {
        if declared != feature_count {
            return Err(sk_shape_mismatch(
                input.name(),
                vec![feature_count],
                vec![declared],
            ));
        }
    }
    Ok(())
}

fn find_bias(model: &SKOnnxModel, matmul_outputs: &[String]) -> Option<SKOnnxConstant> {
    let graph = &model.model().graph;
    let add = graph.nodes.iter().find(|node| {
        node.op_type == "Add"
            && node
                .inputs
                .iter()
                .any(|input| matmul_outputs.contains(input))
    })?;
    add.inputs
        .iter()
        .filter(|input| !matmul_outputs.contains(input))
        .find_map(|input| constant(model, input))
}
