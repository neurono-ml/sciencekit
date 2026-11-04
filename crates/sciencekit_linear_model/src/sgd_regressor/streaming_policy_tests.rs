//! Companion tests for the pass policy, the supported floats and the rejection
//! paths of the multi-epoch out-of-core regression fit
//! (spec `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKReopenableSource};

use super::SKSGDRegressor;
use super::streaming_tests::split_batches;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::test_datasets::{SKSupervisedFloatSource, assert_close_f32, regression_design, to_f32};

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

/// A reopenable factory that fails on the second pass aborts the fit: the
/// partial first-pass iterate is discarded and the original cause surfaces.
#[test]
fn a_reopenable_factory_failure_on_the_second_pass_aborts_the_fit() {
    let (features, targets) = regression_design();
    let mut calls = 0usize;
    let provider = SKReopenableSource::new(move || {
        calls += 1;
        if calls == 2 {
            Err(SKError::Conversion(
                "factory failed on the second pass".into(),
            ))
        } else {
            Ok(SKSupervisedFloatSource::<f64>::new(vec![(
                features.clone(),
                targets.clone(),
            )]))
        }
    });
    let error = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(3)
        .build()
        .unwrap()
        .fit_streaming(provider)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("factory failed on the second pass")
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
