//! Concurrent-prediction acceptance: one fitted model serves several threads.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use crate::classifier::SKDecisionTreeClassifier;

/// Every concurrent prediction equals the sequential result.
#[test]
fn a_large_model_predicts_concurrently() {
    let rows = 20_000usize;
    let features = Arc::new(Array2::from_shape_fn((rows, 4), |(row, column)| {
        ((row * 19 + column * 23) % 89) as f64 / 44.0 - 1.0
    }));
    let labels: Array1<i64> =
        Array1::from_shape_fn(rows, |row| i64::from(features[[row, 0]] > 0.0));
    let model = Arc::new(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_depth(5)
            .build()
            .expect("builds")
            .fit(features.view(), labels.view())
            .expect("fit succeeds"),
    );
    let expected = model.predict_labels(features.view()).expect("predict");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = Arc::clone(&features);
            std::thread::spawn(move || model.predict_labels(features.view()).expect("predict"))
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().expect("thread joins"), expected);
    }
}
