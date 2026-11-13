//! The interpreter adapter on an identity graph.

use ndarray::array;
use sciencekit_common::SKRegressorPredictor;

use super::SKOnnxPredictor;
use super::test_support::identity_model;

#[test]
fn identity_graph_returns_its_input_unchanged() {
    let model = identity_model();
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model)
        .build()
        .expect("the identity model is a valid predictor");
    let features = array![[1.0, 2.0], [3.0, 4.0]];
    let outputs = predictor
        .predict_arrays(features.view())
        .expect("inference runs");
    let identity = outputs.get("out").expect("the identity output exists");
    assert_eq!(identity, &features.to_owned().into_dyn());
}

#[test]
fn identity_graph_also_satisfies_the_regression_trait() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(identity_model())
        .build()
        .expect("the identity model builds");
    let features = array![[1.0, 2.0]];
    let error = predictor
        .predict(features.view())
        .expect_err("width 2 is not a response");
    assert!(matches!(
        error,
        super::SKOnnxImportError::Central(sciencekit_common::SKError::UnsupportedOperation { .. })
    ));
}
