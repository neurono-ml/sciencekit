//! Batch-size behaviour: one row, many rows and row-by-row agreement.

use ndarray::{Array2, array};
use sciencekit_common::{SKClassifierPredictor, SKRegressorPredictor};

use super::SKOnnxPredictor;
use super::test_support::{affine_regressor_model, softmax_classifier_model};

fn batch(rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, 2), |(row, column)| {
        (row as f64) * 0.1 + column as f64
    })
}

#[test]
fn a_single_row_produces_a_single_response() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the model builds");
    let prediction = predictor.predict(array![[1.0, 1.0]].view()).expect("runs");
    assert_eq!(prediction.len(), 1);
}

#[test]
fn a_large_batch_preserves_the_row_count_and_agrees_row_by_row() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(affine_regressor_model())
        .build()
        .expect("the model builds");
    let features = batch(257);
    let prediction = predictor.predict(features.view()).expect("runs");
    assert_eq!(prediction.len(), 257);
    for row in 0..features.nrows() {
        let single = predictor
            .predict(
                features
                    .row(row)
                    .to_owned()
                    .insert_axis(ndarray::Axis(0))
                    .view(),
            )
            .expect("runs");
        assert!((single[0] - prediction[row]).abs() < 1e-12);
    }
}

#[test]
fn classifier_batches_agree_row_by_row() {
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(softmax_classifier_model())
        .build()
        .expect("the model builds");
    let features = batch(64);
    let batch_probabilities = predictor
        .predict_probabilities(features.view())
        .expect("runs");
    for row in 0..features.nrows() {
        let single = predictor
            .predict_probabilities(
                features
                    .row(row)
                    .to_owned()
                    .insert_axis(ndarray::Axis(0))
                    .view(),
            )
            .expect("runs");
        for class in 0..3 {
            assert!((single[[0, class]] - batch_probabilities[[row, class]]).abs() < 1e-12);
        }
    }
}
