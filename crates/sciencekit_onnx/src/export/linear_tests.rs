//! Linear and logistic parity tests against the Rust predictions and the
//! reference ONNX runtime.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::acceptance_tests::{assert_classifier_interface, write_fixture};

use super::SKToOnnx;

fn features<F: sciencekit_common::SKFloat>() -> Array2<F> {
    array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 5.0]
    ]
    .mapv(|value| F::from(value).expect("the fixture is finite"))
}

fn json_matrix<F: sciencekit_common::SKFloat>(matrix: &Array2<F>) -> serde_json::Value {
    serde_json::Value::Array(
        matrix
            .rows()
            .into_iter()
            .map(|row| {
                serde_json::json!(
                    row.iter()
                        .map(|value| value.to_f64().expect("the fixture is finite"))
                        .collect::<Vec<f64>>()
                )
            })
            .collect(),
    )
}

fn json_vector<F: sciencekit_common::SKFloat>(values: &Array1<F>) -> serde_json::Value {
    serde_json::json!(
        values
            .iter()
            .map(|value| value.to_f64().expect("the fixture is finite"))
            .collect::<Vec<f64>>()
    )
}

#[test]
fn linear_regression_export_matches_the_rust_prediction() {
    use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;

    let features: Array2<f64> = features();
    let targets: Array1<f64> = array![1.0, 3.0, 4.0, 14.0, 9.0, 23.0];
    let model = SKLinearRegression::build()
        .build()
        .expect("the regressor configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the regressor fits");
    let prediction = model
        .predict(features.view())
        .expect("the regressor predicts");
    let onnx = model.to_onnx().expect("the regressor exports");
    assert_eq!(onnx.outputs().len(), 1);
    assert_eq!(onnx.outputs()[0].feature_count(), Some(1));
    write_fixture(
        "linear_regression_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f64>>>() }),
        1e-9,
    );
}

#[test]
fn ridge_single_alpha_export_matches_the_rust_prediction() {
    use sciencekit_linear_model::ridge::SKRidge;

    let features: Array2<f32> = features();
    let targets: Array1<f32> = array![1.0, 3.0, 4.0, 14.0, 9.0, 23.0];
    let model = SKRidge::<f32>::build()
        .alpha(0.5_f32)
        .build()
        .expect("the regressor configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the regressor fits");
    let prediction = model
        .predict(features.view())
        .expect("the regressor predicts");
    let onnx = model.to_onnx().expect("the regressor exports");
    write_fixture(
        "ridge_single_alpha_f32",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f32>>>() }),
        1e-5,
    );
}

#[test]
fn logistic_regression_binary_export_matches_the_rust_probabilities() {
    use sciencekit_linear_model::logistic_regression::SKLogisticRegression;

    let features: Array2<f64> = features();
    let targets: Array1<i64> = array![0, 0, 0, 1, 1, 1];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the classifier fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the classifier predicts");
    let labels = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    assert_classifier_interface(&onnx, 2, 2, crate::graph::SKOnnxTensorDType::Double);
    write_fixture(
        "logistic_regression_binary_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            "label": labels.to_vec(),
            "probabilities": json_matrix(&probabilities),
        }),
        1e-7,
    );
}

#[test]
fn logistic_regression_multiclass_export_matches_the_rust_probabilities() {
    use sciencekit_linear_model::logistic_regression::SKLogisticRegression;

    let features: Array2<f64> = array![
        [0.0, 0.0],
        [0.2, 0.1],
        [5.0, 5.0],
        [5.2, 4.8],
        [0.1, 5.0],
        [0.0, 5.2]
    ];
    let targets: Array1<i64> = array![0, 0, 1, 1, 2, 2];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the classifier fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the classifier predicts");
    let labels = model.predict_labels(features.view()).expect("the labels");
    let onnx = model.to_onnx().expect("the classifier exports");
    assert_classifier_interface(&onnx, 2, 3, crate::graph::SKOnnxTensorDType::Double);
    write_fixture(
        "logistic_regression_multiclass_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            "label": labels.to_vec(),
            "probabilities": json_matrix(&probabilities),
        }),
        1e-7,
    );
}

#[test]
fn logistic_regression_export_declares_the_class_table_in_metadata() {
    use sciencekit_linear_model::logistic_regression::SKLogisticRegression;

    let features: Array2<f64> = features();
    let targets: Array1<i64> = array![0, 0, 0, 1, 1, 1];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the classifier fits");
    let onnx = model.to_onnx().expect("the classifier exports");
    assert_eq!(
        onnx.metadata()
            .get("sciencekit.number_of_classes")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        onnx.metadata()
            .get("sciencekit.class_label.1")
            .map(String::as_str),
        Some("1")
    );
    let _ = json_vector(&array![1.0_f64]);
}
