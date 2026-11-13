//! `SKLinearSVC` export: `X · Wᵀ + b` followed by `ArgMax`.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::{floating_tensor, require_feature_width};
use super::super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) fn append_linear_svc<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (shape, coefficients) = floating_tensor(state, "linear_svc.heads.coefficients")?;
    let (_, intercepts) = floating_tensor(state, "linear_svc.heads.intercepts")?;
    let &[heads, feature_count] = shape.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1, state.metadata().feature_count()],
            found: shape,
        });
    };
    require_feature_width(state, feature_count)?;
    let classes = state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or_else(|| if heads == 1 { 2 } else { heads });
    // A single head exposes [-d, d]; multiclass exposes one head per class.
    let weights: Vec<F> = if heads == 1 {
        coefficients
            .iter()
            .map(|value| -*value)
            .chain(coefficients.iter().copied())
            .collect()
    } else {
        coefficients.clone()
    };
    let biases: Vec<F> = if heads == 1 {
        vec![-intercepts[0], intercepts[0]]
    } else {
        intercepts.clone()
    };
    let output_heads = weights.len() / feature_count;
    let mut transposed = vec![F::zero(); weights.len()];
    for head in 0..output_heads {
        for feature in 0..feature_count {
            transposed[feature * output_heads + head] = weights[head * feature_count + feature];
        }
    }
    builder.add_floating_initializer(
        format!("{namespace}linear_svc.weights"),
        vec![feature_count, output_heads],
        &transposed,
    );
    builder.add_floating_initializer(
        format!("{namespace}linear_svc.intercepts"),
        vec![1, output_heads],
        &biases,
    );
    let product = format!("{namespace}linear_svc.product");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![input.to_string(), format!("{namespace}linear_svc.weights")],
        vec![product.clone()],
    ));
    let decision = format!("{namespace}decision_values");
    builder.add_node(SKOnnxNode::new(
        "Add",
        vec![product, format!("{namespace}linear_svc.intercepts")],
        vec![decision.clone()],
    ));
    let label = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new("ArgMax", vec![decision.clone()], vec![label.clone()])
            .attribute_integer("axis", 1)
            .attribute_integer("keepdims", 0),
    );
    Ok(SKOnnxStageOutput {
        tensor: decision.clone(),
        width: classes,
        kind: SKOnnxStageKind::DecisionValues {
            label_tensor: label,
            decision_tensor: decision,
            decision_width: classes,
        },
    })
}
