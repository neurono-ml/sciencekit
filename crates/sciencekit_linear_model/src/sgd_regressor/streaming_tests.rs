//! Companion tests for the single-pass out-of-core regression fit (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKReopenableSource, SKSupervisedFit};

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

/// A one-shot source with a larger budget still takes exactly one pass and does
/// not raise under the strict policy.
#[test]
fn a_one_shot_source_with_a_larger_budget_takes_one_pass() {
    let (features, targets) = regression_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &targets, 2));
    let model = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(5)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(model.epochs_taken(), 1);
    assert!(!model.converged());
}

/// A one-shot source with a budget of exactly one pass really did exhaust its
/// budget, so the strict policy raises instead of pretending the provider
/// clamped it.
#[test]
fn a_one_shot_source_with_a_budget_of_one_raises_under_strict() {
    let (features, targets) = regression_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &targets, 2));
    let error = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .number_of_epochs_without_improvement(10)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 1 }));
}

/// A reopenable source that exhausts the whole requested budget without
/// converging raises under the strict policy.
#[test]
fn an_exhausted_reopenable_budget_raises_under_strict() {
    let (features, targets) = regression_design();
    let batches = split_batches(&features, &targets, 2);
    let provider =
        SKReopenableSource::new(move || Ok(SKSupervisedFloatSource::<f64>::new(batches.clone())));
    let error = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .number_of_epochs_without_improvement(10)
        .build()
        .unwrap()
        .fit_streaming(provider)
        .unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 2 }));
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
