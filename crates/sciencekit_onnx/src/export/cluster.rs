//! Clustering exporters (spec `onnx-export`, requirement "Per-family
//! operators").
//!
//! A fitted k-means model predicts the nearest centre; the graph therefore
//! reduces to a squared-distance matrix and `ArgMin`.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::floating_tensor;
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_KMEANS_NAME: &str = "SKKMeans";
pub(crate) const SK_MINIBATCH_KMEANS_NAME: &str = "SKMiniBatchKMeans";

/// Translate one clustering estimator state, when the name matches.
pub(crate) fn append_cluster<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    if estimator_name != SK_KMEANS_NAME && estimator_name != SK_MINIBATCH_KMEANS_NAME {
        return Ok(None);
    }
    let (center_shape, centers) = floating_tensor(state, "kmeans.cluster_centers")?;
    let &[cluster_count, feature_count] = center_shape.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![state.metadata().feature_count()],
            found: center_shape,
        });
    };
    builder.add_floating_initializer(
        format!("{namespace}kmeans.centers"),
        vec![cluster_count, feature_count],
        &centers,
    );
    builder.add_integer_initializer(format!("{namespace}kmeans.query_axes"), vec![1], &[1]);
    let query = format!("{namespace}kmeans.query");
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![input.to_string(), format!("{namespace}kmeans.query_axes")],
        vec![query.clone()],
    ));
    let reference = format!("{namespace}kmeans.reference");
    builder.add_integer_initializer(format!("{namespace}kmeans.reference_axes"), vec![1], &[0]);
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![
            format!("{namespace}kmeans.centers"),
            format!("{namespace}kmeans.reference_axes"),
        ],
        vec![reference.clone()],
    ));
    let difference = format!("{namespace}kmeans.difference");
    builder.add_node(SKOnnxNode::new(
        "Sub",
        vec![query, reference],
        vec![difference.clone()],
    ));
    let squared = format!("{namespace}kmeans.squared");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![difference.clone(), difference],
        vec![squared.clone()],
    ));
    let distances = format!("{namespace}kmeans.distances");
    builder.add_integer_initializer(format!("{namespace}kmeans.distance_axes"), vec![1], &[2]);
    builder.add_node(
        SKOnnxNode::new(
            "ReduceSum",
            vec![squared, format!("{namespace}kmeans.distance_axes")],
            vec![distances.clone()],
        )
        .attribute_integer("keepdims", 0),
    );
    let label = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new("ArgMin", vec![distances], vec![label.clone()])
            .attribute_integer("axis", 1)
            .attribute_integer("keepdims", 0),
    );
    Ok(Some(SKOnnxStageOutput {
        tensor: label.clone(),
        width: 1,
        kind: SKOnnxStageKind::Labels {
            label_tensor: label,
            label_count: cluster_count,
        },
    }))
}
