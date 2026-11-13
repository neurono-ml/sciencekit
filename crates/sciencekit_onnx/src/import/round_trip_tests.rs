//! Exported library models re-imported and compared with the Rust prediction.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKFeatureTransformer, SKRegressorPredictor, SKSupervisedFit,
    SKUnsupervisedFit,
};

use super::SKOnnxPredictor;
use crate::SKToOnnx;

fn features_f64() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 5.0]
    ]
}

fn features_f32() -> Array2<f32> {
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

#[test]
fn linear_regression_round_trips_through_export_and_import() {
    use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;

    let features = features_f64();
    let targets: Array1<f64> = array![1.0, 3.0, 4.0, 14.0, 9.0, 23.0];
    let model = SKLinearRegression::build()
        .build()
        .expect("the configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the regressor fits");
    let prediction = model.predict(features.view()).expect("the model predicts");
    let imported = import(&model.to_onnx().expect("the model exports"));
    let restored = imported
        .predict(features.view())
        .expect("the import predicts");
    for (expected, actual) in prediction.iter().zip(restored.iter()) {
        assert!((expected - actual).abs() < 1e-5, "{expected} != {actual}");
    }
}

#[test]
fn standard_scaler_round_trips_through_export_and_import() {
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features = features_f64();
    let model = SKStandardScaler::build()
        .with_mean(true)
        .with_std(true)
        .build()
        .expect("the configuration is valid")
        .fit(&features)
        .expect("the scaler fits");
    let transformed = model
        .transform(features.view())
        .expect("the scaler transforms");
    let imported = import(&model.to_onnx().expect("the model exports"));
    let outputs = imported
        .predict_arrays(features.view())
        .expect("the import runs");
    let restored = outputs.values().next().expect("one output");
    let flat: Vec<f64> = restored.iter().copied().collect();
    for (expected, actual) in transformed.iter().zip(flat.iter()) {
        assert!((expected - actual).abs() < 1e-5, "{expected} != {actual}");
    }
}

#[test]
fn decision_tree_classifier_round_trips_through_export_and_import() {
    use sciencekit_tree::classifier::SKDecisionTreeClassifier;

    let features = features_f32();
    let labels = ["a", "a", "a", "a", "b", "b", "b", "b"];
    let model = SKDecisionTreeClassifier::build()
        .maximum_depth(3)
        .build()
        .expect("the configuration is valid")
        .fit(features.view(), labels.as_slice())
        .expect("the tree fits");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("the tree predicts");
    let imported = import(&model.to_onnx().expect("the model exports"));
    let restored = imported
        .predict_probabilities(features.view())
        .expect("the import predicts");
    assert_eq!(probabilities.shape(), restored.shape());
    for (expected, actual) in probabilities.iter().zip(restored.iter()) {
        assert!((expected - actual).abs() < 1e-6, "{expected} != {actual}");
    }
}

#[test]
fn pipeline_regressor_round_trips_through_export_and_import() {
    use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
    use sciencekit_pipeline::SKPipeline;
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features = features_f64();
    let targets: Array1<f64> = array![1.0, 3.0, 4.0, 14.0, 9.0, 23.0];
    let pipeline = SKPipeline::start::<f64>()
        .then(
            SKStandardScaler::build()
                .build()
                .expect("the configuration is valid"),
        )
        .finalize_regressor(
            SKLinearRegression::build()
                .build()
                .expect("the configuration is valid"),
        )
        .expect("the pipeline configuration is valid")
        .fit(features.view(), targets.view())
        .expect("the pipeline fits");
    let prediction = pipeline
        .predict(features.view())
        .expect("the pipeline predicts");
    let imported = import(&pipeline.to_onnx().expect("the pipeline exports"));
    let restored = imported
        .predict(features.view())
        .expect("the import predicts");
    for (expected, actual) in prediction.iter().zip(restored.iter()) {
        assert!((expected - actual).abs() < 1e-5, "{expected} != {actual}");
    }
}

fn import<F: sciencekit_common::SKFloat>(model: &crate::graph::SKOnnxModel) -> SKOnnxPredictor<F> {
    SKOnnxPredictor::<F>::builder()
        .model_bytes(model.bytes().to_vec())
        .build()
        .expect("the exported model imports")
}
