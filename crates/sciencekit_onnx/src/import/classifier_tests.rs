//! Classifier probabilities, labels and head ambiguity.

use ndarray::{Array1, array};
use sciencekit_common::SKClassifierPredictor;

use super::SKOnnxImportError;
use super::SKOnnxPredictor;
use super::test_support::{affine_regressor_model, softmax_classifier_model};

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
