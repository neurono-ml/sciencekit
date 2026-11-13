//! Early-stopping and fitted-iteration tests (task 4.4).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::SKHistGradientBoostingRegressor;

fn flat_data(rows: usize) -> (Array2<f64>, ndarray::Array1<f64>) {
    let features = Array2::from_shape_fn((rows, 1), |(row, _)| row as f64);
    let targets = ndarray::Array1::from_elem(rows, 3.0);
    (features, targets)
}

#[test]
fn hist_early_stop_reports_the_fitted_iteration_count() {
    let (features, targets) = flat_data(100);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(20)
        .n_iter_no_change(2)
        .validation_fraction(0.2)
        .maximum_leaf_nodes(8)
        .minimum_samples_leaf(2)
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    assert!(model.number_of_stages() < 20);
    assert_eq!(model.number_of_stages(), 1);
    assert_eq!(model.initial_prediction(), 3.0);
}

#[test]
fn hist_early_stop_disabled_fits_every_iteration() {
    let (features, targets) = flat_data(60);
    let model = SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(7)
        .without_early_stopping()
        .minimum_samples_leaf(2)
        .build()
        .expect("build")
        .fit(features.view(), targets.view())
        .expect("fit");
    assert_eq!(model.number_of_stages(), 7);
}

#[test]
fn hist_early_stop_keeps_every_stage_when_no_validation_split_is_possible() {
    // Three rows and three classes cannot hold out a stratified validation
    // split; early stopping must then behave as disabled instead of silently
    // keeping only the first stage.
    let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("data");
    let labels = ndarray::Array1::from_vec(vec![0i64, 1, 2]);
    let model = super::SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(5)
        .minimum_samples_leaf(1)
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    assert_eq!(model.number_of_stages(), 5);
}

#[test]
fn hist_early_stop_is_seed_deterministic() {
    let (features, targets) = (
        Array2::from_shape_fn((80, 2), |(row, column)| ((row * 11 + column) % 17) as f64),
        ndarray::Array1::from_shape_fn(80, |row| ((row * 5) % 13) as f64 - 4.0),
    );
    let fit_once = || {
        SKHistGradientBoostingRegressor::<f64>::build()
            .number_of_estimators(30)
            .n_iter_no_change(3)
            .random_seed(11)
            .minimum_samples_leaf(1)
            .maximum_leaf_nodes(4)
            .maximum_bins(8)
            .build()
            .expect("build")
            .fit(features.view(), targets.view())
            .expect("fit")
            .number_of_stages()
    };
    assert_eq!(fit_once(), fit_once());
}
