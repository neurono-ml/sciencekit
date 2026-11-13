//! Pipeline parity tests: stage subgraphs compose in order.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor};

use crate::acceptance_tests::{assert_classifier_interface, write_fixture};

use super::SKToOnnx;

fn features() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.5, 0.2],
        [1.0, 0.0],
        [0.2, 1.0],
        [1.0, 1.0],
        [1.5, 1.2]
    ]
}

fn json_matrix(matrix: &Array2<f64>) -> serde_json::Value {
    serde_json::json!(
        matrix
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>()
    )
}

#[test]
fn export_pipeline_classifier_composes_scaler_and_logistic_regression() {
    use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
    use sciencekit_pipeline::SKPipeline;
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features = features();
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
    let pipeline = SKPipeline::start::<f64>()
        .then(
            SKStandardScaler::build()
                .build()
                .expect("the scaler configuration is valid"),
        )
        .finalize_classifier(
            SKLogisticRegression::build()
                .build()
                .expect("the classifier configuration is valid"),
        )
        .expect("the pipeline configuration is valid");
    let fitted = pipeline
        .fit(&features, labels.view())
        .expect("the pipeline fits");
    let probabilities = fitted
        .predict_probabilities(features.view())
        .expect("the pipeline predicts");
    let predicted = fitted
        .predict_labels(features.view())
        .expect("the pipeline labels");
    let onnx = fitted.to_onnx().expect("the pipeline exports");
    assert_classifier_interface(&onnx, 2, 2, crate::graph::SKOnnxTensorDType::Double);
    write_fixture(
        "pipeline_classifier_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({
            onnx.outputs()[0].name(): predicted.to_vec(),
            onnx.outputs()[1].name(): json_matrix(&probabilities),
        }),
        1e-7,
    );
}

#[test]
fn export_pipeline_regressor_composes_two_transforms_and_a_linear_model() {
    use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
    use sciencekit_pipeline::SKPipeline;
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features = features();
    let targets: Array1<f64> = array![1.0, 1.4, 2.0, 1.6, 3.0, 3.1];
    let pipeline = SKPipeline::start::<f64>()
        .then(
            SKStandardScaler::build()
                .build()
                .expect("the scaler configuration is valid"),
        )
        .finalize_regressor(
            SKLinearRegression::build()
                .build()
                .expect("the regressor configuration is valid"),
        )
        .expect("the pipeline configuration is valid");
    let fitted = pipeline
        .fit(&features, targets.view())
        .expect("the pipeline fits");
    let prediction = fitted
        .predict(features.view())
        .expect("the pipeline predicts");
    let onnx = fitted.to_onnx().expect("the pipeline exports");
    write_fixture(
        "pipeline_regressor_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): prediction.iter().map(|value| vec![*value]).collect::<Vec<Vec<f64>>>() }),
        1e-9,
    );
}
