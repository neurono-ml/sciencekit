//! Companion tests: hostile ONNX graphs are rejected with a typed error before
//! any native model is built (spec `native-model-import`, requirement
//! "Validation and concurrency").

use sciencekit_interop::native::SKNativeImportError;

use super::SKOnnxNativeError;
use super::SKOnnxNativeImporter;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

fn affine_builder(
    input_element_type: SKOnnxTensorDType,
    weight: impl FnOnce(&mut SKOnnxGraphBuilder),
    feature_count: usize,
) -> SKOnnxGraphBuilder {
    let mut builder = SKOnnxGraphBuilder::new("hostile_affine");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", input_element_type, feature_count).unwrap(),
        )
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "weight".to_string()],
            vec!["prediction".to_string()],
        ))
        .add_output(SKOnnxValueInfo::floating_matrix("prediction", input_element_type, 1).unwrap());
    weight(&mut builder);
    builder
}

fn import_error(builder: SKOnnxGraphBuilder) -> SKOnnxNativeError {
    SKOnnxNativeImporter::builder()
        .model(builder.build().unwrap())
        .build::<f64>()
        .unwrap_err()
}

/// The declared input element type must match the initializer element type.
#[test]
fn native_onnx_rejects_an_input_dtype_that_disagrees_with_the_initializer() {
    let error = import_error(affine_builder(
        SKOnnxTensorDType::Double,
        |builder| {
            builder.add_floating_initializer("weight", vec![2, 1], &[0.5_f32, 1.5]);
        },
        2,
    ));
    assert!(matches!(
        error,
        SKOnnxNativeError::Native(SKNativeImportError::DtypeMismatch { ref tensor, .. })
            if tensor == "X"
    ));
}

/// An integer affine weight is not a floating model parameter.
#[test]
fn native_onnx_rejects_integer_affine_weights() {
    let error = import_error(affine_builder(
        SKOnnxTensorDType::Double,
        |builder| {
            builder.add_integer_initializer("weight", vec![2, 1], &[1, 2]);
        },
        2,
    ));
    assert!(matches!(
        error,
        SKOnnxNativeError::Native(SKNativeImportError::DtypeMismatch { ref tensor, .. })
            if tensor == "weight"
    ));
}

/// The declared input width must match the weight matrix.
#[test]
fn native_onnx_rejects_an_input_width_that_disagrees_with_the_weights() {
    let error = import_error(affine_builder(
        SKOnnxTensorDType::Double,
        |builder| {
            builder.add_floating_initializer("weight", vec![3, 1], &[0.5_f64, 1.5, 2.5]);
        },
        2,
    ));
    assert!(matches!(
        error,
        SKOnnxNativeError::Native(SKNativeImportError::ShapeMismatch { ref tensor, .. })
            if tensor == "X"
    ));
}

/// A weight matrix with zero features cannot yield a usable model.
#[test]
fn native_onnx_rejects_a_zero_column_weight() {
    let error = import_error(affine_builder(
        SKOnnxTensorDType::Double,
        |builder| {
            builder.add_floating_initializer::<f64>("weight", vec![0, 1], &[]);
        },
        0,
    ));
    assert!(matches!(
        error,
        SKOnnxNativeError::Native(SKNativeImportError::ShapeMismatch { ref tensor, .. })
            if tensor == "feature_count"
    ));
}

/// An integer scaler statistic is not a floating learned parameter.
#[test]
fn native_onnx_rejects_integer_scaler_statistics() {
    let mut builder = SKOnnxGraphBuilder::new("hostile_scaler");
    builder
        .add_input(SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 1).unwrap())
        .add_integer_initializer("standard_scaler.mean", vec![1], &[4])
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["X".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Double, 1).unwrap(),
        )
        .add_metadata("sciencekit.estimator_name", "SKStandardScaler")
        .add_metadata("sciencekit.feature_count", "1");
    let error = import_error(builder);
    assert!(matches!(
        error,
        SKOnnxNativeError::Native(SKNativeImportError::DtypeMismatch { ref tensor, .. })
            if tensor == "standard_scaler.mean"
    ));
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
