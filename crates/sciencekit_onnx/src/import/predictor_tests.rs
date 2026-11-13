//! Regression and classifier predictor behaviour.

use ndarray::{Array1, array};
use sciencekit_common::{SKClassifierPredictor, SKRegressorPredictor};

use super::SKOnnxImportError;
use super::SKOnnxPredictor;
use super::test_support::{affine_regressor_model, softmax_classifier_model};

#[test]
fn affine_regressor_matches_the_closed_form() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the regressor model builds");
    let features = array![[1.0, 2.0], [0.0, 0.0]];
    let prediction = predictor.predict(features.view()).expect("inference runs");
    assert!((prediction[0] - 0.5).abs() < 1e-12);
    assert!((prediction[1] - 0.5).abs() < 1e-12);
}

#[test]
fn feature_width_mismatch_is_rejected_before_inference() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the regressor model builds");
    let error = predictor
        .predict(array![[1.0, 2.0, 3.0]].view())
        .expect_err("must reject");
    assert!(matches!(error, SKOnnxImportError::ShapeMismatch { .. }));
}

#[test]
fn non_finite_input_is_rejected() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the regressor model builds");
    let error = predictor
        .predict(array![[f64::NAN, 0.0]].view())
        .expect_err("must reject");
    assert!(matches!(
        error,
        SKOnnxImportError::Central(sciencekit_common::SKError::NonFiniteValue { .. })
    ));
}

#[test]
fn f64_predictor_explicitly_converts_to_an_f32_model_input() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(super::test_support::identity_model_f32())
        .build()
        .expect("the model builds");
    let features = array![[1.0_f64, 2.0]];
    let outputs = predictor
        .predict_arrays(features.view())
        .expect("inference runs");
    assert_eq!(outputs["out"].shape(), &[1, 2]);
}

#[test]
fn softmax_classifier_exposes_labels_and_probabilities() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(softmax_classifier_model())
        .build()
        .expect("the classifier model builds");
    let features = array![[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
    let labels = predictor
        .predict_labels(features.view())
        .expect("labels run");
    assert_eq!(labels, Array1::from_vec(vec![0, 1, 0]));
    let probabilities = predictor
        .predict_probabilities(features.view())
        .expect("probabilities run");
    assert_eq!(probabilities.shape(), &[3, 3]);
    for row in probabilities.rows() {
        let total: f64 = row.sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}

#[test]
fn regression_head_rejects_classifier_methods() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the regressor model builds");
    let error = predictor
        .predict_labels(array![[1.0, 2.0]].view())
        .expect_err("a regressor has no labels");
    assert!(matches!(
        error,
        SKOnnxImportError::Central(sciencekit_common::SKError::UnsupportedOperation { .. })
    ));
}

#[test]
fn two_float_outputs_are_an_ambiguous_head() {
    let mut exported = affine_regressor_model();
    let mut graph = exported.model().graph.clone();
    graph.outputs.push(graph.outputs[0].clone());
    let model = serde_onnx::ir::Model::new(graph, exported.model().opset_import.clone());
    exported = crate::graph::SKOnnxModel::from_parts_in_tests(
        serde_onnx::proto::encode_model(&model),
        model,
    );
    let error = SKOnnxPredictor::<f64>::builder()
        .model(exported)
        .build()
        .expect_err("two float outputs are ambiguous");
    assert!(matches!(error, SKOnnxImportError::UnsupportedOutput { .. }));
}
