//! Tree and forest parity tests: ONNX-ML tree ensembles against the Rust
//! predictions and the reference runtime.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::acceptance_tests::{assert_classifier_interface, write_fixture};

use super::SKToOnnx;

fn features() -> Array2<f32> {
    array![
        [0.0, 0.0],
        [0.5, 0.2],
        [1.0, 0.0],
        [0.2, 1.0],
        [1.0, 1.0],
        [1.5, 1.2],
        [2.0, 2.0],
        [2.2, 1.8]
    ]
}

fn json_matrix(matrix: &Array2<f32>) -> serde_json::Value {
    serde_json::json!(
        matrix
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f32>>>()
    )
}

#[test]
fn decision_tree_classifier_export_matches_the_rust_probabilities() {
    use sciencekit_tree::classifier::SKDecisionTreeClassifier;

    let features = features();
    let labels = ["a", "a", "a", "a", "b", "b", "b", "b"];
    let model = SKDecisionTreeClassifier::build()
        .maximum_depth(3)
        .build()
        .expect("the tree configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the tree fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the tree predicts");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the tree exports");
    assert_classifier_interface(&onnx, 2, 2, crate::graph::SKOnnxTensorDType::Float);
    assert_eq!(
        onnx.metadata()
            .get("sciencekit.estimator_name")
            .map(String::as_str),
        Some("SKDecisionTreeClassifier")
    );
    write_fixture(
        "decision_tree_classifier_f32",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            "label": predicted.to_vec(),
            "probabilities": json_matrix(&probabilities),
        }),
        1e-5,
    );
}

#[test]
fn decision_tree_regressor_export_matches_the_rust_prediction() {
    use sciencekit_tree::regressor::SKDecisionTreeRegressor;

    let features = features();
    let targets = array![1.0_f32, 1.5, 2.0, 1.2, 3.0, 3.4, 5.0, 4.8];
    let model = SKDecisionTreeRegressor::build()
        .maximum_depth(3)
        .build()
        .expect("the tree configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the tree fits");
    let prediction = model.predict(features.view()).expect("the tree predicts");
    let onnx = model.to_onnx().expect("the tree exports");
    write_fixture(
        "decision_tree_regressor_f32",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f32>>>() }),
        1e-5,
    );
}

#[test]
fn random_forest_classifier_export_matches_the_rust_probabilities() {
    use sciencekit_ensemble::random_forest::SKRandomForestClassifier;

    let features = features();
    let labels = ["a", "a", "a", "a", "b", "b", "b", "b"];
    let model = SKRandomForestClassifier::<f32>::build()
        .number_of_trees(5)
        .maximum_depth(3)
        .random_seed(11)
        .build()
        .expect("the forest configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the forest fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the forest predicts");
    let predicted = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the forest exports");
    assert_classifier_interface(&onnx, 2, 2, crate::graph::SKOnnxTensorDType::Float);
    write_fixture(
        "random_forest_classifier_f32",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            "label": predicted.to_vec(),
            "probabilities": json_matrix(&probabilities),
        }),
        1e-5,
    );
}

#[test]
fn random_forest_regressor_export_matches_the_rust_prediction() {
    use sciencekit_ensemble::random_forest::SKRandomForestRegressor;

    let features = features();
    let targets = array![1.0_f32, 1.5, 2.0, 1.2, 3.0, 3.4, 5.0, 4.8];
    let model = SKRandomForestRegressor::<f32>::build()
        .number_of_trees(5)
        .maximum_depth(3)
        .random_seed(11)
        .build()
        .expect("the forest configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the forest fits");
    let prediction = model.predict(features.view()).expect("the forest predicts");
    let onnx = model.to_onnx().expect("the forest exports");
    write_fixture(
        "random_forest_regressor_f32",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f32>>>() }),
        1e-5,
    );
}

#[test]
fn f64_trees_are_rejected_because_the_onnx_ml_attributes_are_floats() {
    use sciencekit_tree::regressor::SKDecisionTreeRegressor;

    let features: Array2<f64> = features().mapv(|value| value as f64);
    let targets = array![1.0_f64, 1.5, 2.0, 1.2, 3.0, 3.4, 5.0, 4.8];
    let model = SKDecisionTreeRegressor::build()
        .maximum_depth(2)
        .build()
        .expect("the tree configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the tree fits");
    let error = model.to_onnx().expect_err("double trees are rejected");
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedOperation {
            operation: "onnx_export",
            ..
        }
    ));
}
