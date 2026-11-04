//! Companion tests for the single-pass out-of-core regression fit (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::test_datasets::{
    SKSupervisedFloatSource, assert_close, assert_close_f32, regression_design, to_f32,
};

fn split_batches<F: sciencekit_common::SKFloat>(
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

#[test]
fn an_f32_stream_fits() {
    let (features, targets) = regression_design();
    let features = to_f32(&features);
    let targets = targets.mapv(|value| value as f32);
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &targets, 2));
    let model = SKSGDRegressor::<f32>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(model.epochs_taken(), 1);
    assert!(model.coefficients().iter().all(|value| value.is_finite()));
    assert_close_f32(model.coefficients().as_slice().unwrap(), &[3.0, 2.0], 0.5);
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

#[test]
fn an_empty_source_is_rejected() {
    let mut source = SKSupervisedFloatSource::<f64>::new(Vec::new());
    let error = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

#[test]
fn a_column_mismatch_is_rejected() {
    let batches = vec![
        (
            Array2::from_elem((2, 1), 1.0),
            Array1::from_vec(vec![0.0, 1.0]),
        ),
        (
            Array2::from_elem((2, 2), 1.0),
            Array1::from_vec(vec![0.0, 1.0]),
        ),
    ];
    let mut source = SKSupervisedFloatSource::new(batches);
    let error = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn early_stopping_over_a_stream_is_refused() {
    let (features, targets) = regression_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &targets, 2));
    let error = SKSGDRegressor::<f64>::build()
        .early_stopping(true)
        .maximum_number_of_iterations(2)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(
        matches!(error, SKError::UnsupportedOperation { operation, .. } if operation == "early_stopping")
    );
}
