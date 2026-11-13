//! Mapping an ONNX-ML tree ensemble to tree/forest state
//! (spec `native-model-import`, requirement "ONNX native import").
use sciencekit_common::{SKFloat, SKModelState, SKParameterValue};
use serde_onnx::ml::{OnnxOp, TreeEnsembleClassifier, TreeEnsembleRegressor};

use super::super::metadata::{SKOnnxNativeMetadata, feature_count};
use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;

use super::encode;
use super::rebuild;

/// Try to map a tree-ensemble node; `None` when the graph carries none.
pub(crate) fn state_from_trees<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: Option<&SKOnnxNativeMetadata>,
) -> Result<Option<SKModelState<F>>, SKOnnxNativeError> {
    let graph = &model.model().graph;
    let Some(node) = graph.nodes.iter().find(|node| {
        matches!(
            node.op_type.as_str(),
            "TreeEnsembleRegressor" | "TreeEnsembleClassifier"
        )
    }) else {
        return Ok(None);
    };
    let features = metadata
        .map(|metadata| metadata.feature_count)
        .or_else(|| feature_count(model))
        .ok_or_else(|| invalid("the graph declares no fixed feature width"))?;
    if node.op_type == "TreeEnsembleClassifier" {
        return classifier_state(metadata, node, features);
    }
    let ensemble =
        TreeEnsembleRegressor::from_node(node).map_err(|error| invalid(error.to_string()))?;
    rebuild::validate_semantics(&ensemble.nodes, "TreeEnsembleRegressor")?;
    regressor_state(metadata, ensemble, features)
}

fn regressor_state<F: SKFloat>(
    metadata: Option<&SKOnnxNativeMetadata>,
    ensemble: TreeEnsembleRegressor,
    features: usize,
) -> Result<Option<SKModelState<F>>, SKOnnxNativeError> {
    let weights = ensemble.target_weights.clone().unwrap_or_default();
    let trees = rebuild::rebuild(&ensemble.nodes, |node_id, tree_id| {
        vec![rebuild::weight_at(
            &ensemble.target_treeids,
            &ensemble.target_nodeids,
            &weights,
            tree_id,
            node_id,
        )]
    });
    let is_forest = metadata.is_some_and(|metadata| metadata.estimator_name.contains("Forest"))
        || trees.len() > 1;
    if is_forest {
        return encode::forest_regressor::<F>(&trees, features).map(Some);
    }
    let tree = trees
        .first()
        .ok_or_else(|| invalid("the ensemble has no trees"))?;
    let metadata = encode::tree_metadata::<F>("SKDecisionTreeRegressor", features)?
        .with_parameter("criterion", SKParameterValue::from("squared_error"))
        .map_err(native)?;
    encode::single_tree(tree, features, metadata).map(Some)
}

fn classifier_state<F: SKFloat>(
    metadata: Option<&SKOnnxNativeMetadata>,
    node: &serde_onnx::ir::Node,
    features: usize,
) -> Result<Option<SKModelState<F>>, SKOnnxNativeError> {
    let ensemble =
        TreeEnsembleClassifier::from_node(node).map_err(|error| invalid(error.to_string()))?;
    rebuild::validate_semantics(&ensemble.nodes, "TreeEnsembleClassifier")?;
    if rebuild::distinct_tree_ids(&ensemble.nodes).len() > 1 {
        return Err(SKOnnxNativeError::UnsupportedOperator {
            operator: "TreeEnsembleClassifier".to_string(),
            reason: "a forest classifier native import needs per-tree local class tables; use the generic predictor or a Safetensors state".to_string(),
        });
    }
    let labels: Vec<String> = metadata
        .and_then(|metadata| metadata.class_labels.clone())
        .or_else(|| {
            ensemble
                .classlabels_int64s
                .as_ref()
                .map(|labels| labels.iter().map(i64::to_string).collect())
        })
        .unwrap_or_else(|| vec!["0".to_string(), "1".to_string()]);
    let class_count = labels.len();
    let weights = ensemble.class_weights.clone().unwrap_or_default();
    let trees = rebuild::rebuild(&ensemble.nodes, |node_id, tree_id| {
        let mut values = vec![0.0_f64; class_count];
        for (position, class) in ensemble.class_ids.iter().enumerate() {
            if ensemble.class_treeids[position] == tree_id
                && ensemble.class_nodeids[position] == node_id
                && (*class as usize) < class_count
            {
                values[*class as usize] += f64::from(weights[position]);
            }
        }
        values
    });
    let tree = trees
        .first()
        .ok_or_else(|| invalid("the ensemble has no trees"))?;
    let metadata = encode::tree_metadata::<F>("SKDecisionTreeClassifier", features)?
        .with_class_labels(labels)
        .map_err(native)?
        .with_parameter(
            "number_of_classes",
            SKParameterValue::from(class_count as i64),
        )
        .map_err(native)?;
    encode::single_tree(tree, features, metadata).map(Some)
}

fn invalid(detail: impl Into<String>) -> SKOnnxNativeError {
    SKOnnxNativeError::InvalidGraph {
        detail: detail.into(),
    }
}

fn native(error: sciencekit_common::SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(sciencekit_interop::native::SKNativeImportError::from(error))
}
