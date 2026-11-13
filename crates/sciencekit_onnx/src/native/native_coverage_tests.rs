//! Companion tests: unsupported operators and the documented ONNX subset
//! (spec `native-model-import`, requirement "ONNX native import").

use ndarray::array;
use sciencekit_common::{SKBuilder, SKSupervisedFit};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;

use super::SKOnnxNativeImporter;
use super::error::SKOnnxNativeError;
use crate::export::SKToOnnx;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

fn exported_linear() -> SKLinearRegressionModel<f64> {
    let features = array![[0.0_f64], [1.0], [2.0]];
    let targets = array![1.0_f64, 3.0, 5.0];
    SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
}

#[test]
fn native_onnx_rejects_an_unsupported_operator() {
    let model = exported_linear();
    let mut builder = SKOnnxGraphBuilder::new("unsupported");
    builder
        .add_input(SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 1).unwrap())
        .add_floating_initializer("weight", vec![1, 1], &[2.0_f64])
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "weight".to_string()],
            vec!["product".to_string()],
        ))
        .add_node(SKOnnxNode::new(
            "Log",
            vec!["product".to_string()],
            vec!["prediction".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("prediction", SKOnnxTensorDType::Double, 1).unwrap(),
        );
    let graph = builder.build().unwrap();
    let error = SKOnnxNativeImporter::builder()
        .model(graph)
        .build::<f64>()
        .unwrap_err();
    assert!(matches!(
        error,
        SKOnnxNativeError::UnsupportedOperator { ref operator, .. } if operator == "Log"
    ));
    let _ = model;
}

#[test]
fn native_onnx_rejects_a_graph_without_a_mappable_structure() {
    let mut builder = SKOnnxGraphBuilder::new("identity");
    builder
        .add_input(SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 1).unwrap())
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["X".to_string()],
            vec!["prediction".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("prediction", SKOnnxTensorDType::Double, 1).unwrap(),
        );
    let error = SKOnnxNativeImporter::builder()
        .model(builder.build().unwrap())
        .build::<f64>()
        .unwrap_err();
    assert!(matches!(
        error,
        SKOnnxNativeError::UnsupportedOperator { .. }
    ));
}

#[test]
fn native_onnx_records_the_requested_backend() {
    let model = exported_linear();
    let imported = SKOnnxNativeImporter::builder()
        .model(model.to_onnx().unwrap())
        .backend(sciencekit_common::SKBackendKind::MatrixMultiply)
        .build::<f64>()
        .unwrap();
    assert_eq!(
        imported.descriptor().backend(),
        sciencekit_common::SKBackendKind::MatrixMultiply
    );
    assert!(imported.supports_retraining());
}
