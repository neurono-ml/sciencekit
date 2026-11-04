//! Large-data acceptance test for the SGD regressor (spec
//! `stochastic-gradient-regressor`, acceptance criteria).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKRegressorPredictor, SKSupervisedFit};

use super::SKSGDRegressor;
use crate::sgd::SKSGDLearningRate;
use crate::test_datasets::{assert_close, to_csr};

const ROWS: usize = 5_000;
const COLUMNS: usize = 30;

fn large_design() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        ((row * 31 + column * 17) % 101) as f64 / 101.0 - 0.5
    });
    let true_coefficients = Array1::from_iter((0..COLUMNS).map(|column| (column % 3) as f64 - 1.0));
    let targets = features.dot(&true_coefficients).mapv(|value| value + 2.5);
    (features, targets, true_coefficients)
}

fn estimator() -> SKSGDRegressor<f64> {
    SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.25)
        .alpha(0.0)
        .shuffle(false)
        .tolerance(1e-9)
        .number_of_epochs_without_improvement(5)
        .maximum_number_of_iterations(200)
        .build()
        .unwrap()
}

#[test]
fn a_large_design_converges_and_agrees_across_representations() {
    let (features, targets, true_coefficients) = large_design();
    let dense = estimator().fit(features.view(), targets.view()).unwrap();
    assert!(dense.converged());
    assert!(dense.epochs_taken() <= 200);
    assert_close(
        dense.coefficients().as_slice().unwrap(),
        true_coefficients.as_slice().unwrap(),
        0.01,
    );

    let sparse = to_csr(&features);
    let sparse_model = estimator().fit(sparse.view(), targets.view()).unwrap();
    let dense_predicted = dense.predict(features.view()).unwrap();
    let sparse_predicted = sparse_model.predict(features.view()).unwrap();
    assert_close(
        dense_predicted.as_slice().unwrap(),
        sparse_predicted.as_slice().unwrap(),
        1e-8,
    );
}
