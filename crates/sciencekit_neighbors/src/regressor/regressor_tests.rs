//! Companion tests for regressor fit and prediction
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKRegressorPredictor, SKSupervisedFit};

use super::SKKNeighborsRegressor;
use crate::neighbors::SKNeighborsWeights;

fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1e-12,
        "expected {right}, found {left}"
    );
}

#[test]
fn regressor_uniform_prediction_is_the_neighbor_mean() {
    let features = array![[1.0_f64], [2.0], [3.0]];
    let responses = array![2.0_f64, 4.0, 6.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[2.0_f64]].view()).unwrap();

    assert_close(predictions[0], 4.0);
}

#[test]
fn regressor_uses_only_the_nearest_neighbors() {
    let features = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let responses = array![1.0_f64, 2.0, 30.0, 40.0, 50.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[1.1_f64]].view()).unwrap();

    assert_close(predictions[0], 1.5);
}

#[test]
fn regressor_distance_weighted_prediction_matches_the_hand_computed_value() {
    let features = array![[1.0_f64], [3.0]];
    let responses = array![10.0_f64, 20.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[0.0_f64]].view()).unwrap();

    // weights 1 and 1/3 → (10 + 20/3) / (4/3)
    assert_close(predictions[0], 12.5);
}

#[test]
fn regressor_zero_distance_neighbors_dominate_the_average() {
    let features = array![[1.0_f64], [1.0], [4.0]];
    let responses = array![10.0_f64, 20.0, 100.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[1.0_f64]].view()).unwrap();

    assert_close(predictions[0], 15.0);
}

#[test]
fn regressor_predicts_one_response_per_query_row() {
    let features = array![[1.0_f64], [10.0]];
    let responses = array![5.0_f64, 50.0];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model
        .predict(array![[1.5_f64], [9.0], [10.0]].view())
        .unwrap();

    assert_eq!(predictions.len(), 3);
    assert_close(predictions[0], 5.0);
    assert_close(predictions[1], 50.0);
    assert_close(predictions[2], 50.0);
}

#[test]
fn regressor_elevates_integer_targets() {
    let features = array![[1.0_f64], [2.0], [3.0]];
    let responses = array![2_i64, 4, 6];
    let model = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(array![[2.0_f64]].view()).unwrap();

    assert_close(predictions[0], 4.0);
}

#[test]
fn regressor_fit_rejects_nominal_targets() {
    let features = Array2::from_shape_fn((4, 1), |(row, _)| row as f64);
    let labels = ["left", "right", "left", "right"];
    let outcome = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), &labels[..]);

    match outcome {
        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
            assert_eq!(representation, "nominal");
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}

#[test]
fn regressor_fit_rejects_a_target_length_mismatch() {
    let features = array![[1.0_f64], [2.0], [3.0]];
    let responses = Array1::from_elem(2, 1.0_f64);
    let outcome = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), responses.view());

    match outcome {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![3, 1]);
            assert_eq!(found, vec![2, 1]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}
