//! Concurrency acceptance for the neighbor estimators
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::classifier::SKKNeighborsClassifierModel;
use crate::neighbors::SKNeighborsWeights;
use crate::regressor::SKKNeighborsRegressorModel;
use crate::{SKKNeighborsClassifier, SKKNeighborsRegressor};

const ROWS: usize = 2_500;

fn assert_send_sync<T: Send + Sync>() {}

fn design() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, 4), |(row, column)| {
        ((row * 17 + column * 31) % 53) as f64 / 7.0
    })
}

fn labels() -> Array1<i64> {
    Array1::from_shape_fn(ROWS, |row| (row % 3) as i64)
}

fn responses() -> Array1<f64> {
    Array1::from_shape_fn(ROWS, |row| (row % 11) as f64 / 2.0)
}

#[test]
fn concurrency_fitted_models_are_send_and_sync() {
    assert_send_sync::<SKKNeighborsClassifierModel<f64>>();
    assert_send_sync::<SKKNeighborsRegressorModel<f64>>();
    assert_send_sync::<SKKNeighborsClassifierModel<f32>>();
    assert_send_sync::<SKKNeighborsRegressorModel<f32>>();
}

#[test]
fn concurrency_one_classifier_serves_several_threads() {
    let data = design();
    let model = SKKNeighborsClassifier::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(data.view(), labels().view())
        .unwrap();
    let expected = model.predict_labels(data.view()).unwrap();
    let batch = 100;

    std::thread::scope(|scope| {
        for chunk in 0..4 {
            let model = &model;
            let data = data
                .slice_axis(
                    ndarray::Axis(0),
                    ndarray::Slice::from(chunk * batch..(chunk + 1) * batch),
                )
                .to_owned();
            let expected = expected
                .slice_axis(
                    ndarray::Axis(0),
                    ndarray::Slice::from(chunk * batch..(chunk + 1) * batch),
                )
                .to_owned();
            scope.spawn(move || {
                let found = model.predict_labels(data.view()).unwrap();
                assert_eq!(found, expected);
            });
        }
    });
}

#[test]
fn concurrency_one_regressor_serves_several_threads() {
    let data = design();
    let model = SKKNeighborsRegressor::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(data.view(), responses().view())
        .unwrap();
    let expected = model.predict(data.view()).unwrap();
    let batch = 100;

    std::thread::scope(|scope| {
        for chunk in 0..4 {
            let model = &model;
            let data = data
                .slice_axis(
                    ndarray::Axis(0),
                    ndarray::Slice::from(chunk * batch..(chunk + 1) * batch),
                )
                .to_owned();
            let expected = expected
                .slice_axis(
                    ndarray::Axis(0),
                    ndarray::Slice::from(chunk * batch..(chunk + 1) * batch),
                )
                .to_owned();
            scope.spawn(move || {
                let found = model.predict(data.view()).unwrap();
                assert_eq!(found, expected);
            });
        }
    });
}
