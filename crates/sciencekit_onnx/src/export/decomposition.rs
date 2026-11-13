//! Decomposition exporters (spec `onnx-export`, requirement "Per-family
//! operators").
//!
//! A fitted PCA or truncated SVD keeps its components and (for PCA) the
//! centring mean; `transform` is therefore `(X - μ) · Wᵀ`, optionally divided by
//! the whitening scale.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::{boolean_parameter, floating_tensor};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_PCA_NAME: &str = "SKPCA";
pub(crate) const SK_TRUNCATED_SVD_NAME: &str = "SKTruncatedSVD";

/// Translate one decomposition estimator state, when the name matches.
pub(crate) fn append_decomposition<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    match estimator_name {
        SK_PCA_NAME => append_pca(state, builder, input, namespace).map(Some),
        SK_TRUNCATED_SVD_NAME => {
            append_projection(state, builder, input, namespace, "truncated_svd", false).map(Some)
        }
        _ => Ok(None),
    }
}

fn append_pca<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let output = append_projection(state, builder, input, namespace, "pca", true)?;
    if boolean_parameter(state, "whiten", false) {
        let (_, whiten_scale) = floating_tensor(state, "pca.whiten_scale")?;
        builder.add_floating_initializer(
            format!("{namespace}pca.whiten_scale"),
            vec![whiten_scale.len()],
            &whiten_scale,
        );
        let scaled = format!("{namespace}pca.whitened");
        builder.add_node(SKOnnxNode::new(
            "Div",
            vec![
                output.tensor.clone(),
                format!("{namespace}pca.whiten_scale"),
            ],
            vec![scaled.clone()],
        ));
        return Ok(SKOnnxStageOutput {
            tensor: scaled,
            width: output.width,
            kind: SKOnnxStageKind::Transformed,
        });
    }
    Ok(output)
}

/// Append `(X - mean) · componentsᵀ` for a family prefix.
fn append_projection<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
    prefix: &str,
    with_mean: bool,
) -> Result<SKOnnxStageOutput, SKError> {
    let (component_shape, components) = floating_tensor(state, &format!("{prefix}.components"))?;
    let &[component_count, feature_count] = component_shape.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![state.metadata().feature_count()],
            found: component_shape,
        });
    };
    let mut transposed = vec![F::zero(); components.len()];
    for component in 0..component_count {
        for feature in 0..feature_count {
            transposed[feature * component_count + component] =
                components[component * feature_count + feature];
        }
    }
    builder.add_floating_initializer(
        format!("{namespace}{prefix}.components"),
        vec![feature_count, component_count],
        &transposed,
    );
    let mut current = input.to_string();
    if with_mean {
        let (_, mean) = floating_tensor(state, &format!("{prefix}.mean"))?;
        builder.add_floating_initializer(
            format!("{namespace}{prefix}.mean"),
            vec![mean.len()],
            &mean,
        );
        let centered = format!("{namespace}{prefix}.centered");
        builder.add_node(SKOnnxNode::new(
            "Sub",
            vec![current, format!("{namespace}{prefix}.mean")],
            vec![centered.clone()],
        ));
        current = centered;
    }
    let projected = format!("{namespace}{prefix}.projected");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![current, format!("{namespace}{prefix}.components")],
        vec![projected.clone()],
    ));
    Ok(SKOnnxStageOutput {
        tensor: projected,
        width: component_count,
        kind: SKOnnxStageKind::Transformed,
    })
}
