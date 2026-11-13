//! Neighbour exporters (spec `onnx-export`, requirement "Per-family
//! operators").
//!
//! The fitted search keeps its training design, so the graph reproduces the
//! brute-force search: pairwise squared distances, `TopK` neighbours and a
//! reduction of the gathered training labels or responses.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::{
    floating_tensor, integer_parameter, integer_tensor, string_parameter, unsupported,
};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_KNN_CLASSIFIER_NAME: &str = "SKKNeighborsClassifier";
pub(crate) const SK_KNN_REGRESSOR_NAME: &str = "SKKNeighborsRegressor";

/// Translate one neighbour estimator state, when the name matches.
pub(crate) fn append_neighbors<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    if estimator_name != SK_KNN_CLASSIFIER_NAME && estimator_name != SK_KNN_REGRESSOR_NAME {
        return Ok(None);
    }
    if string_parameter(state, "training_representation", "dense") != "dense" {
        return Err(unsupported(
            estimator_name,
            "the exported graph recomputes distances from a dense training design; a CSR state is not translated",
        ));
    }
    if string_parameter(state, "weights", "uniform") != "uniform" {
        return Err(unsupported(
            estimator_name,
            "only uniform neighbour weights are exported; distance weighting needs a different reduction",
        ));
    }
    let (training_shape, training_values) = floating_tensor(state, "knn.training_dense")?;
    let &[training_rows, feature_count] = training_shape.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![state.metadata().feature_count()],
            found: training_shape,
        });
    };
    let neighbors = integer_parameter(state, "nearest_neighbors_count", 1).max(1) as usize;
    // Pairwise squared distances through broadcasting: [N, 1, F] - [1, M, F].
    builder.add_floating_initializer(
        format!("{namespace}knn.training"),
        vec![training_rows, feature_count],
        &training_values,
    );
    builder.add_integer_initializer(format!("{namespace}knn.query_axes"), vec![1], &[1]);
    let query = format!("{namespace}knn.query");
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![input.to_string(), format!("{namespace}knn.query_axes")],
        vec![query.clone()],
    ));
    let reference = format!("{namespace}knn.reference");
    builder.add_integer_initializer(format!("{namespace}knn.reference_axes"), vec![1], &[0]);
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![
            format!("{namespace}knn.training"),
            format!("{namespace}knn.reference_axes"),
        ],
        vec![reference.clone()],
    ));
    let difference = format!("{namespace}knn.difference");
    builder.add_node(SKOnnxNode::new(
        "Sub",
        vec![query, reference],
        vec![difference.clone()],
    ));
    let squared = format!("{namespace}knn.squared");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![difference.clone(), difference],
        vec![squared.clone()],
    ));
    let distances = format!("{namespace}knn.distances");
    builder.add_integer_initializer(format!("{namespace}knn.distance_axes"), vec![1], &[2]);
    builder.add_node(
        SKOnnxNode::new(
            "ReduceSum",
            vec![squared, format!("{namespace}knn.distance_axes")],
            vec![distances.clone()],
        )
        .attribute_integer("keepdims", 0),
    );
    let negated = format!("{namespace}knn.negated_distances");
    builder.add_node(SKOnnxNode::new(
        "Neg",
        vec![distances],
        vec![negated.clone()],
    ));
    builder.add_integer_initializer(
        format!("{namespace}knn.count"),
        vec![1],
        &[neighbors as i64],
    );
    let neighbor_values = format!("{namespace}knn.neighbor_values");
    let neighbor_indices = format!("{namespace}knn.neighbor_indices");
    builder.add_node(
        SKOnnxNode::new(
            "TopK",
            vec![negated, format!("{namespace}knn.count")],
            vec![neighbor_values.clone(), neighbor_indices.clone()],
        )
        .attribute_integer("axis", 1)
        .attribute_integer("largest", 1)
        .attribute_integer("sorted", 1),
    );
    let _ = neighbor_values;
    if estimator_name == SK_KNN_REGRESSOR_NAME {
        let (_, responses) = floating_tensor(state, "knn.responses")?;
        builder.add_floating_initializer(
            format!("{namespace}knn.responses"),
            vec![responses.len(), 1],
            &responses,
        );
        let gathered = format!("{namespace}knn.gathered_responses");
        builder.add_node(
            SKOnnxNode::new(
                "Gather",
                vec![format!("{namespace}knn.responses"), neighbor_indices],
                vec![gathered.clone()],
            )
            .attribute_integer("axis", 0),
        );
        let output = format!("{namespace}prediction");
        builder.add_node(
            SKOnnxNode::new("ReduceMean", vec![gathered], vec![output.clone()])
                .attribute_integers("axes", vec![1])
                .attribute_integer("keepdims", 0),
        );
        return Ok(Some(SKOnnxStageOutput {
            tensor: output,
            width: 1,
            kind: SKOnnxStageKind::Regression,
        }));
    }
    let class_indices = integer_tensor(state, "knn.class_indices")?;
    let classes = state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or_else(|| class_indices.iter().copied().max().unwrap_or(0).max(0) as usize + 1);
    builder.add_integer_initializer(
        format!("{namespace}knn.class_indices"),
        vec![class_indices.len()],
        &class_indices,
    );
    let gathered = format!("{namespace}knn.gathered_classes");
    builder.add_node(
        SKOnnxNode::new(
            "Gather",
            vec![format!("{namespace}knn.class_indices"), neighbor_indices],
            vec![gathered.clone()],
        )
        .attribute_integer("axis", 0),
    );
    builder.add_integer_initializer(format!("{namespace}knn.count_axes"), vec![1], &[1]);
    let mut per_class = Vec::new();
    for class in 0..classes {
        builder.add_integer_initializer(
            format!("{namespace}knn.class_code.{class}"),
            Vec::new(),
            &[class as i64],
        );
        let matched = format!("{namespace}knn.class_matched.{class}");
        builder.add_node(SKOnnxNode::new(
            "Equal",
            vec![
                gathered.clone(),
                format!("{namespace}knn.class_code.{class}"),
            ],
            vec![matched.clone()],
        ));
        let indicator = format!("{namespace}knn.class_indicator.{class}");
        builder.add_node(
            SKOnnxNode::new("Cast", vec![matched], vec![indicator.clone()])
                .attribute_integer("to", super::helpers::element_type::<F>()),
        );
        let count = format!("{namespace}knn.class_count.{class}");
        builder.add_node(
            SKOnnxNode::new(
                "ReduceSum",
                vec![indicator, format!("{namespace}knn.count_axes")],
                vec![count.clone()],
            )
            .attribute_integer("keepdims", 1),
        );
        per_class.push(count);
    }
    let counts = format!("{namespace}knn.counts");
    builder.add_node(
        SKOnnxNode::new("Concat", per_class, vec![counts.clone()]).attribute_integer("axis", 1),
    );
    let probabilities = format!("{namespace}probabilities");
    builder.add_floating_initializer(
        format!("{namespace}knn.neighbor_scale"),
        Vec::new(),
        &[F::from(neighbors).ok_or(SKError::Conversion(
            "the neighbour count does not fit the scalar".to_string(),
        ))?],
    );
    builder.add_node(SKOnnxNode::new(
        "Div",
        vec![counts, format!("{namespace}knn.neighbor_scale")],
        vec![probabilities.clone()],
    ));
    let label = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new("ArgMax", vec![probabilities.clone()], vec![label.clone()])
            .attribute_integer("axis", 1)
            .attribute_integer("keepdims", 0),
    );
    Ok(Some(SKOnnxStageOutput {
        tensor: probabilities.clone(),
        width: classes,
        kind: SKOnnxStageKind::Classification {
            label_tensor: label,
            probability_tensor: probabilities,
            class_count: classes,
        },
    }))
}
