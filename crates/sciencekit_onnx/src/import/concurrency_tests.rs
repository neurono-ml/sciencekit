//! A loaded predictor is shareable and deterministic under concurrency.

use std::sync::Arc;
use std::thread;

use ndarray::array;
use sciencekit_common::SKRegressorPredictor;

use super::SKOnnxPredictor;
use super::test_support::affine_regressor_model;

#[test]
fn the_predictor_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKOnnxPredictor<f64>>();
}

#[test]
fn concurrent_predictions_match_the_sequential_result() {
    let predictor = Arc::new(
        SKOnnxPredictor::<f64>::builder()
            .model(affine_regressor_model())
            .build()
            .expect("the model builds"),
    );
    let features = Arc::new(array![[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]);
    let sequential = predictor.predict(features.view()).expect("runs");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let predictor = Arc::clone(&predictor);
            let features = Arc::clone(&features);
            thread::spawn(move || predictor.predict(features.view()).expect("runs"))
        })
        .collect();
    for handle in handles {
        let concurrent = handle.join().expect("the thread does not panic");
        assert_eq!(concurrent, sequential);
    }
}
