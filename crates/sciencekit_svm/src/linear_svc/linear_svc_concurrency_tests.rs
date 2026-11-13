//! Companion TDD tests for determinism and concurrent prediction
//! (spec `linear-support-vector-classifier`).

use std::sync::Arc;
use std::thread;

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::{SKLinearSVC, SKLinearSVCModel};

fn assert_send_sync<T: Send + Sync>() {}

fn fixture() -> (ndarray::Array2<f64>, ndarray::Array1<i64>) {
    let features = array![
        [-2.5_f64, -2.0],
        [-1.5, -1.0],
        [-1.0, -1.5],
        [0.0, 0.5],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.5, 2.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1, 1];
    (features, labels)
}

#[test]
fn repeated_fits_produce_identical_models() {
    let (features, labels) = fixture();
    let fit_once = || {
        SKLinearSVC::<f64>::build()
            .C(0.7)
            .maximum_number_of_iterations(500)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };
    let first = fit_once();
    let second = fit_once();

    assert_eq!(first.heads().len(), second.heads().len());
    first
        .heads()
        .iter()
        .zip(second.heads().iter())
        .for_each(|(left, right)| {
            assert_eq!(left.coefficients(), right.coefficients());
            assert_eq!(left.intercept(), right.intercept());
            assert_eq!(left.iterations(), right.iterations());
        });
    assert_eq!(
        first.predict_labels(features.view()).unwrap(),
        second.predict_labels(features.view()).unwrap()
    );
}

#[test]
fn concurrent_prediction_matches_the_sequential_result() {
    let (features, labels) = fixture();
    let model = Arc::new(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap(),
    );
    let sequential = model.predict_labels(features.view()).unwrap();
    let sequential_decisions = model.decision_function(features.view()).unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            thread::spawn(move || {
                (
                    model.predict_labels(features.view()).unwrap(),
                    model.decision_function(features.view()).unwrap(),
                )
            })
        })
        .collect();

    handles.into_iter().for_each(|handle| {
        let (labels, decisions) = handle.join().expect("prediction thread panicked");
        assert_eq!(labels, sequential);
        assert_eq!(decisions, sequential_decisions);
    });
}

#[test]
fn estimator_and_fitted_model_are_send_and_sync() {
    assert_send_sync::<SKLinearSVC<f64>>();
    assert_send_sync::<SKLinearSVC<f32>>();
    assert_send_sync::<SKLinearSVCModel<f64>>();
    assert_send_sync::<SKLinearSVCModel<f32>>();
}
