//! Metric-production acceptance through `sciencekit_metrics`
//! (PRD §8.7, spec `linear-support-vector-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};

use crate::SKLinearSVC;

#[test]
fn acceptance_predictions_feed_accuracy_and_macro_f1() {
    let rows = 90usize;
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        let class = row / 30;
        let center = match class {
            0 => [-3.0, 0.0, 1.0],
            1 => [0.0, 3.0, -1.0],
            _ => [3.0, 0.0, 0.5],
        };
        center[column] + ((row * 7 + column * 5) % 11) as f64 * 0.01
    });
    let labels = Array1::from_shape_fn(rows, |row| (row / 30) as i64);

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predicted = model.predict_labels(features.view()).unwrap();
    let truth = labels.to_vec();
    let predicted = predicted.to_vec();

    let accuracy: f64 = sk_accuracy_score(&truth, &predicted).unwrap();
    let macro_f1: f64 = sk_f1_score(&truth, &predicted, None, SKAveraging::Macro).unwrap();
    assert!(accuracy > 0.99, "accuracy was {accuracy}");
    assert!(macro_f1 > 0.99, "macro F1 was {macro_f1}");
}
