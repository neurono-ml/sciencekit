//! Companion tests: hostile ONNX graphs are rejected with a typed error before
//! any native model is built (spec `native-model-import`, requirement
//! "Validation and concurrency").

use super::SKOnnxNativeError;
use super::SKOnnxNativeImporter;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

fn import_error(builder: SKOnnxGraphBuilder) -> SKOnnxNativeError {
    SKOnnxNativeImporter::builder()
        .model(builder.build().unwrap())
        .build::<f64>()
        .unwrap_err()
}

fn tree_node(mode: &str, missing_value_tracks_true: i64) -> SKOnnxNode {
    SKOnnxNode::ml(
        "TreeEnsembleRegressor",
        vec!["X".to_string()],
        vec!["prediction".to_string()],
    )
    .attribute_integer("n_targets", 1)
    .attribute_integers("nodes_treeids", vec![0, 0, 0])
    .attribute_integers("nodes_nodeids", vec![0, 1, 2])
    .attribute_integers("nodes_featureids", vec![0, 0, 0])
    .attribute_strings("nodes_modes", vec![mode, "LEAF", "LEAF"])
    .attribute_floats("nodes_values", vec![0.5, 0.0, 0.0])
    .attribute_integers("nodes_truenodeids", vec![1, 0, 0])
    .attribute_integers("nodes_falsenodeids", vec![2, 0, 0])
    .attribute_integers(
        "nodes_missing_value_tracks_true",
        vec![missing_value_tracks_true, 0, 0],
    )
    .attribute_integers("target_treeids", vec![0, 0])
    .attribute_integers("target_nodeids", vec![1, 2])
    .attribute_integers("target_ids", vec![0, 0])
    .attribute_floats("target_weights", vec![10.0, 20.0])
}

fn tree_error(node: SKOnnxNode) -> SKOnnxNativeError {
    let mut builder = SKOnnxGraphBuilder::new("hostile_tree");
    builder
        .add_input(SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Float, 2).unwrap())
        .add_node(node)
        .add_output(
            SKOnnxValueInfo::floating_matrix("prediction", SKOnnxTensorDType::Float, 1).unwrap(),
        );
    import_error(builder)
}

/// The native tree only routes with `feature <= threshold` and finite input.
/// A graph whose branch mode or missing-value rule differs would silently
/// predict different values, so it is rejected instead of flattened.
#[test]
fn native_onnx_rejects_tree_branch_modes_it_cannot_represent() {
    for mode in ["BRANCH_GT", "BRANCH_LT", "BRANCH_GTE", "BRANCH_EQ"] {
        let error = tree_error(tree_node(mode, 0));
        assert!(
            matches!(
                error,
                SKOnnxNativeError::UnsupportedOperator { ref operator, .. }
                    if operator == "TreeEnsembleRegressor"
            ),
            "mode {mode} must be rejected"
        );
    }
    let error = tree_error(tree_node("BRANCH_LEQ", 1));
    assert!(matches!(
        error,
        SKOnnxNativeError::UnsupportedOperator { ref operator, .. }
            if operator == "TreeEnsembleRegressor"
    ));
}
