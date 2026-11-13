//! Repeated-fit determinism, thread-count invariance and concurrent prediction
//! tests (task 5.3).

use std::sync::Arc;
use std::thread;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};

use super::SKHistGradientBoostingRegressor;

fn data() -> (Array2<f64>, ndarray::Array1<f64>) {
    let rows = 120usize;
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        ((row * 13 + column * 7) % 29) as f64 / 4.0
    });
    let targets =
        ndarray::Array1::from_shape_fn(rows, |row| features[[row, 0]] * 0.5 + features[[row, 2]]);
    (features, targets)
}

fn estimator(seed: u64) -> SKHistGradientBoostingRegressor<f64> {
    SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(10)
        .learning_rate(0.3)
        .maximum_leaf_nodes(5)
        .minimum_samples_leaf(1)
        .maximum_bins(16)
        .random_seed(seed)
        .without_early_stopping()
        .build()
        .expect("build")
}

#[test]
fn hist_determinism_repeated_fits_agree() {
    let (features, targets) = data();
    let first = estimator(7)
        .fit(features.view(), targets.view())
        .expect("fit");
    let second = estimator(7)
        .fit(features.view(), targets.view())
        .expect("fit");
    assert_eq!(first.number_of_stages(), second.number_of_stages());
    (0..first.number_of_stages()).for_each(|stage| {
        assert_eq!(
            first.stage_tree(stage).expect("tree"),
            second.stage_tree(stage).expect("tree")
        );
    });
    let first_predictions = first.predict(features.view()).expect("predict");
    let second_predictions = second.predict(features.view()).expect("predict");
    assert_eq!(first_predictions, second_predictions);
}

#[test]
fn hist_determinism_concurrent_prediction_agrees() {
    let (features, targets) = data();
    let model = Arc::new(
        estimator(3)
            .fit(features.view(), targets.view())
            .expect("fit"),
    );
    let sequential = model.predict(features.view()).expect("predict");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = features.clone();
            thread::spawn(move || model.predict(features.view()).expect("predict"))
        })
        .collect();
    handles.into_iter().for_each(|handle| {
        let concurrent = handle.join().expect("thread");
        assert_eq!(concurrent, sequential);
    });
}

#[test]
fn hist_determinism_model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<super::SKHistGradientBoostingRegressorModel<f64>>();
    assert_send_sync::<super::SKHistGradientBoostingClassifierModel<f64>>();
}

#[test]
fn hist_determinism_is_thread_count_invariant() {
    let (features, targets) = data();
    let baseline = estimator(5)
        .fit(features.view(), targets.view())
        .expect("fit");
    let sequential = baseline.predict(features.view()).expect("predict");
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool")
        .install(|| {
            estimator(5)
                .fit(features.view(), targets.view())
                .expect("fit")
        });
    let parallel_predictions = parallel.predict(features.view()).expect("predict");
    assert_eq!(sequential, parallel_predictions);
    (0..baseline.number_of_stages()).for_each(|stage| {
        assert_eq!(
            baseline.stage_tree(stage).expect("tree"),
            parallel.stage_tree(stage).expect("tree")
        );
    });
}
