//! Companion tests: ONNX graphs become native models with matching predictions
//! (spec `native-model-import`, requirement "ONNX native import").

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKFeatureTransformer, SKRegressorPredictor, SKSupervisedFit,
    SKUnsupervisedFit,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sciencekit_tree::SKDecisionTreeRegressor;

use super::SKOnnxNativeImporter;
use super::error::SKOnnxNativeError;
use crate::export::SKToOnnx;
use crate::import::SKOnnxPredictor;

fn features() -> ndarray::Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
    ]
}

fn targets() -> ndarray::Array1<f64> {
    array![1.0_f64, 3.0, 4.0, 14.0, 11.0, 16.0]
}

#[test]
fn native_onnx_imports_an_exported_linear_regression() {
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features().view(), targets().view())
        .unwrap();
    let onnx = model.to_onnx().unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(onnx)
        .build::<f64>()
        .unwrap();
    assert_eq!(
        model.predict(features().view()).unwrap(),
        imported.predict_regression(features().view()).unwrap()
    );
    assert_eq!(imported.descriptor().estimator_name(), "SKLinearRegression");
}

#[test]
fn native_onnx_imports_an_exported_logistic_regression() {
    let labels = ["low", "low", "low", "high", "high", "high"];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features().view(), labels.as_slice())
        .unwrap();
    let onnx = model.to_onnx().unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(onnx)
        .build::<f64>()
        .unwrap();
    assert_eq!(
        model.predict_labels(features().view()).unwrap(),
        imported.predict_labels(features().view()).unwrap()
    );
    assert_eq!(
        model.predict_probabilities(features().view()).unwrap(),
        imported.predict_probabilities(features().view()).unwrap()
    );
    let mut labels = imported.descriptor().class_labels().unwrap().to_vec();
    labels.sort();
    assert_eq!(labels, ["high", "low"]);
}

#[test]
fn native_onnx_imports_an_exported_standard_scaler() {
    let model = SKStandardScaler::build()
        .build()
        .unwrap()
        .fit(features().view())
        .unwrap();
    let onnx = model.to_onnx().unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(onnx)
        .build::<f64>()
        .unwrap();
    assert_eq!(
        model.transform(features().view()).unwrap(),
        imported.transform(features().view()).unwrap()
    );
}

#[test]
fn native_onnx_imports_an_exported_decision_tree() {
    let features = features().mapv(|value| value as f32);
    let targets = targets().mapv(|value| value as f32);
    let model = SKDecisionTreeRegressor::<f32>::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let onnx = model.to_onnx().unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(onnx)
        .build::<f32>()
        .unwrap();
    assert_eq!(
        model.predict(features.view()).unwrap(),
        imported.predict_regression(features.view()).unwrap()
    );
    assert_eq!(
        imported.descriptor().estimator_name(),
        "SKDecisionTreeRegressor"
    );
}

fn external_affine_graph() -> crate::graph::SKOnnxModel {
    use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};
    let mut builder = SKOnnxGraphBuilder::new("external_affine");
    builder
        .add_input(SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2).unwrap())
        .add_floating_initializer("coefficients", vec![2, 1], &[2.0_f64, -1.0])
        .add_floating_initializer("intercept", vec![1, 1], &[0.5_f64])
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "coefficients".to_string()],
            vec!["product".to_string()],
        ))
        .add_node(SKOnnxNode::new(
            "Add",
            vec!["product".to_string(), "intercept".to_string()],
            vec!["prediction".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("prediction", SKOnnxTensorDType::Double, 1).unwrap(),
        );
    builder.build().unwrap()
}

#[test]
fn native_onnx_imports_an_external_affine_graph() {
    let model = external_affine_graph();
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model.clone())
        .build()
        .unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(model)
        .build::<f64>()
        .unwrap();
    let generic = predictor
        .predict_arrays(features().view())
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    let native = imported.predict_regression(features().view()).unwrap();
    assert_eq!(
        generic.iter().copied().collect::<Vec<f64>>(),
        native.iter().copied().collect::<Vec<f64>>()
    );
    assert_eq!(imported.descriptor().estimator_name(), "SKLinearRegression");
}

#[test]
fn native_onnx_rejects_a_forest_classifier_graph() {
    use sciencekit_ensemble::random_forest::SKRandomForestClassifier;
    let labels = ["a", "a", "a", "b", "b", "b"];
    let features = features().mapv(|value| value as f32);
    let model = SKRandomForestClassifier::<f32>::build()
        .number_of_trees(3)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features.view(), labels.as_slice())
        .unwrap();
    let error = SKOnnxNativeImporter::builder()
        .model(model.to_onnx().unwrap())
        .build::<f32>()
        .unwrap_err();
    assert!(matches!(
        error,
        SKOnnxNativeError::UnsupportedOperator { ref operator, .. }
            if operator == "TreeEnsembleClassifier"
    ));
}
