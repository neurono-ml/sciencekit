//! Thread-safety and float-coverage tests for the fitted lasso model
//! (spec `lasso`).

use crate::coordinate_descent::test_datasets::orthogonal_design;
use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKRegressorPredictor, SKSupervisedFit};

use super::core_implementation::SKLasso;

/// One fitted model serves concurrent predictions with identical results.
#[test]
fn model_serves_concurrent_predictions() {
    use std::sync::Arc;
    use std::thread;

    let (features, targets) = orthogonal_design();
    let model = Arc::new(
        SKLasso::build()
            .alpha(0.5)
            .tolerance(1e-10)
            .maximum_number_of_iterations(100_000)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap(),
    );
    let expected = model.predict(features.view()).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            let expected = expected.clone();
            thread::spawn(move || {
                let predictions = model.predict(features.view()).unwrap();
                assert_eq!(predictions, expected);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

/// The estimator fits both supported floats.
#[test]
fn fits_both_supported_floats() {
    let features = array![[1.0f32], [2.0], [3.0]];
    let targets = array![2.0f32, 4.0, 6.0];
    let model = SKLasso::<f32>::build()
        .alpha(1.0f32)
        .include_intercept(false)
        .tolerance(1e-5)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!((model.coefficient_for(0)[0] - 25.0 / 14.0).abs() < 1e-3);
}

/// The fitted model is `Send + Sync`.
#[test]
fn model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::model::SKLassoModel<f64>>();
}
