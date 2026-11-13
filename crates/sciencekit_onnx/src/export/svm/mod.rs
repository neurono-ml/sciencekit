//! Support-vector-machine exporters (spec `onnx-export`, requirement
//! "Per-family operators").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classification;
mod expansion;
mod kernel;

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::{floating_tensor, integer_parameter, unsupported};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_SVC_NAME: &str = "SKSVC";
pub(crate) const SK_SVR_NAME: &str = "SKSVR";
pub(crate) const SK_LINEAR_SVC_NAME: &str = "SKLinearSVC";

/// Translate one support-vector estimator state, when the name matches.
pub(crate) fn append_svm<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    match estimator_name {
        SK_SVR_NAME => append_svr(state, builder, input, namespace).map(Some),
        SK_SVC_NAME => append_svc(state, builder, input, namespace).map(Some),
        SK_LINEAR_SVC_NAME => append_linear_svc(state, builder, input, namespace).map(Some),
        _ => Ok(None),
    }
}

/// Read a dense support-vector block, rejecting a CSR block explicitly.
fn dense_support_vectors<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    estimator_name: &str,
) -> Result<(Vec<usize>, Vec<F>), SKError> {
    let tensor_name = format!("{prefix}.support_vectors_dense");
    if state.tensor(&tensor_name).is_none() {
        return Err(unsupported(
            estimator_name,
            "the support-vector block is stored as CSR; the exported graph needs a dense support-vector matrix",
        ));
    }
    expansion::read_dense_support_vectors(state, prefix)
}

fn append_svr<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (shape, support_vectors) = dense_support_vectors(state, "svr", SK_SVR_NAME)?;
    builder.add_floating_initializer(
        format!("{namespace}svr.support_vectors"),
        shape,
        &support_vectors,
    );
    let (_, dual_coefficients) = floating_tensor(state, "svr.dual_coefficients")?;
    let bias = expansion::read_bias(state, "svr")?;
    let margin = expansion::append_expansion(
        state,
        builder,
        input,
        &format!("{namespace}svr.support_vectors"),
        &dual_coefficients,
        bias,
        namespace,
    )?;
    Ok(SKOnnxStageOutput {
        tensor: margin,
        width: 1,
        kind: SKOnnxStageKind::Regression,
    })
}

fn append_svc<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let pair_count = integer_parameter(state, "number_of_pairs", 0).max(0) as usize;
    if pair_count == 0 {
        return Err(unsupported(
            SK_SVC_NAME,
            "the state carries no pairwise head",
        ));
    }
    let mut margins = Vec::with_capacity(pair_count);
    let mut pairs = Vec::with_capacity(pair_count);
    for index in 0..pair_count {
        let prefix = format!("svc.pair.{index}");
        let (shape, support_vectors) = dense_support_vectors(state, &prefix, SK_SVC_NAME)?;
        builder.add_floating_initializer(
            format!("{namespace}svc.pair.{index}.support_vectors"),
            shape,
            &support_vectors,
        );
        let (_, dual_coefficients) =
            floating_tensor(state, &format!("{prefix}.dual_coefficients"))?;
        let bias = expansion::read_bias(state, &prefix)?;
        let margin = expansion::append_expansion(
            state,
            builder,
            input,
            &format!("{namespace}svc.pair.{index}.support_vectors"),
            &dual_coefficients,
            bias,
            &format!("{namespace}svc.pair.{index}."),
        )?;
        margins.push(margin);
        pairs.push((
            integer_parameter(state, &format!("{prefix}.class_negative"), 0).max(0) as usize,
            integer_parameter(state, &format!("{prefix}.class_positive"), 1).max(0) as usize,
        ));
    }
    let classes = state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or_else(|| {
            pairs
                .iter()
                .map(|(_, positive)| positive + 1)
                .max()
                .unwrap_or(2)
        });
    if pair_count == 1 {
        let margin = margins.pop().expect("one pair");
        let polarity = classification::concat_polarity(builder, &margin, namespace);
        let label = format!("{namespace}label");
        builder.add_node(
            SKOnnxNode::new("ArgMax", vec![polarity], vec![label.clone()])
                .attribute_integer("axis", 1)
                .attribute_integer("keepdims", 0),
        );
        return Ok(SKOnnxStageOutput {
            tensor: margin.clone(),
            width: 1,
            kind: SKOnnxStageKind::DecisionValues {
                label_tensor: label,
                decision_tensor: margin,
                decision_width: 1,
            },
        });
    }
    let (votes, decision) =
        classification::append_one_vs_one::<F>(builder, &margins, &pairs, classes, namespace)?;
    let label = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new("ArgMax", vec![votes], vec![label.clone()])
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

fn append_linear_svc<F: SKFloat>(
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
