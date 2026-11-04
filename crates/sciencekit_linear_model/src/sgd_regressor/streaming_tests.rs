//! Companion tests for the multi-epoch out-of-core regression fit over source
//! providers (spec `stochastic-gradient-regressor`).
//!
//! This sibling owns the shared split helpers and the tests comparing the
//! streamed trajectory with the in-memory fit.

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKReopenableSource, SKSupervisedFit};

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::test_datasets::{SKSupervisedFloatSource, assert_close, regression_design};

pub(super) fn split_batches<F: sciencekit_common::SKFloat>(
    features: &Array2<F>,
    targets: &Array1<F>,
    split: usize,
) -> Vec<(Array2<F>, Array1<F>)> {
    vec![
        (
            features.slice(ndarray::s![0..split, ..]).to_owned(),
            targets.slice(ndarray::s![0..split]).to_owned(),
        ),
        (
            features.slice(ndarray::s![split.., ..]).to_owned(),
            targets.slice(ndarray::s![split..]).to_owned(),
        ),
    ]
}

fn permissive_single_epoch() -> SKSGDRegressor<f64> {
    SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
}

#[test]
fn one_streaming_pass_equals_one_in_memory_epoch() {
    let (features, targets) = regression_design();
    let in_memory = permissive_single_epoch()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &targets, 2));
    let streamed = permissive_single_epoch()
        .fit_streaming(&mut source)
        .unwrap();
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close(&[in_memory.intercept()], &[streamed.intercept()], 1e-12);
    assert_eq!(streamed.epochs_taken(), 1);
    assert!(!streamed.converged());
}

#[test]
fn a_single_batch_stream_equals_one_epoch() {
    let (features, targets) = regression_design();
    let in_memory = permissive_single_epoch()
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = SKSupervisedFloatSource::new(vec![(features.clone(), targets.clone())]);
    let streamed = permissive_single_epoch()
        .fit_streaming(&mut source)
        .unwrap();
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-12,
    );
}

/// A reopenable provider runs the same epoch budget as the in-memory fit: with
/// a fixed seed and one batch per pass the streamed trajectory is identical.
#[test]
fn a_reopenable_source_runs_the_epoch_budget_and_matches_the_in_memory_fit() {
    let (features, targets) = regression_design();
    let estimator = || {
        SKSGDRegressor::<f64>::build()
            .learning_rate(SKSGDLearningRate::Constant)
            .initial_learning_rate(0.05)
            .alpha(0.0)
            .shuffle(true)
            .random_seed(11)
            .maximum_number_of_iterations(3)
            .average(true)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
    };
    let in_memory = estimator().fit(features.view(), targets.view()).unwrap();
    let batches = vec![(features.clone(), targets.clone())];
    let provider =
        SKReopenableSource::new(move || Ok(SKSupervisedFloatSource::<f64>::new(batches.clone())));
    let streamed = estimator().fit_streaming(provider).unwrap();
    assert_eq!(streamed.epochs_taken(), 3);
    assert!(!streamed.converged());
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close(&[in_memory.intercept()], &[streamed.intercept()], 1e-12);
}

#[test]
fn a_many_batch_stream_is_consumed_in_one_pass() {
    // The streaming driver binds one batch at a time and never stores a
    // trajectory, so a long source must be exactly one in-memory epoch in the
    // same order. Retention is a structural property (the driver holds only the
    // current batch); this test observes the resulting one-pass contract.
    let rows = 64;
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        ((row * (column + 3)) % 17) as f64 / 17.0
    });
    let targets = Array1::from_shape_fn(rows, |row| {
        let x0 = ((row * 3) % 17) as f64 / 17.0;
        let x1 = ((row * 4) % 17) as f64 / 17.0;
        3.0 * x0 + 2.0 * x1
    });
    let in_memory = permissive_single_epoch()
        .fit(features.view(), targets.view())
        .unwrap();
    let batches = (0..rows)
        .map(|row| {
            (
                features.slice(ndarray::s![row..row + 1, ..]).to_owned(),
                targets.slice(ndarray::s![row..row + 1]).to_owned(),
            )
        })
        .collect();
    let mut source = SKSupervisedFloatSource::new(batches);
    let streamed = permissive_single_epoch()
        .fit_streaming(&mut source)
        .unwrap();
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-9,
    );
    assert_close(&[in_memory.intercept()], &[streamed.intercept()], 1e-9);
    assert_eq!(streamed.epochs_taken(), 1);
    assert!(!streamed.converged());
}
