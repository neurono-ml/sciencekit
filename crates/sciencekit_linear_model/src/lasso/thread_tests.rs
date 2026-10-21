//! Thread-safety tests for the fitted lasso model (spec `lasso`).

use crate::coordinate_descent::test_datasets::orthogonal_design;
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
