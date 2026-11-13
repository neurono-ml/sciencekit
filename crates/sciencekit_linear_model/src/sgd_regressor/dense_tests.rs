//! Companion tests for the dense squared-error fit of the SGD regressor (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};

use super::{SKSGDRegressor, SKSGDRegressorCoefficients};
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDRegressorLoss};
use crate::test_datasets::{
    assert_close, assert_close_f32, assert_close_scalar, regression_design, to_f32,
};

#[test]
fn one_update_equals_the_closed_form_squared_error_step() {
    let features = array![[1.0, 2.0]];
    let targets = array![5.0];
    let learning_rate = 0.1;
    let model = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(learning_rate)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    // slope = prediction − target = −5, so w = η·5·x and b = η·5.
    assert_close(model.coefficients().as_slice().unwrap(), &[0.5, 1.0], 1e-12);
    assert_close_scalar(model.intercept(), 0.5, 1e-12);
}

#[test]
fn prediction_matches_the_linear_predictor() {
    let (features, _) = regression_design();
    let coefficients = Array1::from_vec(vec![1.5, -2.0]);
    let intercept = 0.75;
    let source = SKSGDRegressorCoefficients::new(coefficients.view(), intercept);
    let targets = Array1::<f64>::zeros(features.nrows());
    let model = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::EpsilonInsensitive)
        .epsilon(1e9)
        .alpha(0.0)
        .warm_start_from(source)
        .shuffle(false)
        .maximum_number_of_iterations(3)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let predicted = model.predict(features.view()).unwrap();
    let expected = features.dot(&coefficients).mapv(|value| value + intercept);
    assert_close(
        predicted.as_slice().unwrap(),
        expected.as_slice().unwrap(),
        1e-12,
    );
}

#[test]
fn a_prediction_shape_mismatch_is_rejected() {
    let (features, targets) = regression_design();
    let model = SKSGDRegressor::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let wrong = Array2::<f64>::zeros((2, 3));
    let error = model.predict(wrong.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn integer_targets_are_promoted() {
    let (features, targets) = regression_design();
    let integers = Array1::from_vec(targets.iter().map(|&value| value as i64).collect());
    let model = SKSGDRegressor::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), integers.view())
        .unwrap();
    let predicted = model.predict(features.view()).unwrap();
    assert!(predicted.iter().all(|value| value.is_finite()));
    let continuous = SKSGDRegressor::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_close(
        predicted.as_slice().unwrap(),
        continuous
            .predict(features.view())
            .unwrap()
            .as_slice()
            .unwrap(),
        1e-12,
    );
}

#[test]
fn nominal_targets_are_rejected() {
    let (features, _) = regression_design();
    let labels = ["a", "b", "c", "d", "e"];
    let error = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), &labels[..])
        .unwrap_err();
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

#[test]
fn a_target_count_mismatch_is_rejected() {
    let (features, _) = regression_design();
    let short = Array1::from_vec(vec![1.0, 2.0]);
    let error = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), short.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_converged_fit_reports_its_epochs() {
    let (features, targets) = regression_design();
    let model = SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .tolerance(1e-12)
        .maximum_number_of_iterations(20_000)
        .number_of_epochs_without_improvement(3)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(model.converged());
    assert!(model.epochs_taken() < 20_000);
}

#[test]
fn the_policy_decides_what_an_exhausted_budget_does() {
    let (features, targets) = regression_design();
    let strict = SKSGDRegressor::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap_err();
    assert!(matches!(strict, SKError::NotConverged { iterations } if iterations == 2));

    let permissive = SKSGDRegressor::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(!permissive.converged());
    assert_eq!(permissive.epochs_taken(), 2);
}

#[test]
fn an_f32_fit_predicts() {
    let (features, targets) = regression_design();
    let features = to_f32(&features);
    let targets = targets.mapv(|value| value as f32);
    let model = SKSGDRegressor::<f32>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(false)
        .tolerance(1e-6)
        .maximum_number_of_iterations(5_000)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let predicted = model.predict(features.view()).unwrap();
    assert!(predicted.iter().all(|value| value.is_finite()));
    assert_close_f32(
        predicted.as_slice().unwrap(),
        targets.as_slice().unwrap(),
        0.05,
    );
}

#[test]
fn the_shared_generator_reproduces_the_sgd_epoch_order() {
    // Cross-crate guarantee (spec `deterministic-randomness`): the shared
    // generator seeded with the estimator's seed yields exactly the visiting
    // order the SGD fit applies, so one seeded epoch replays bit-for-bit.
    use sciencekit_common::SKRandomGenerator;

    use crate::sgd::config::SKSGDConfig;
    use crate::sgd::{sk_regressor_loss_adapter, sk_run_dense_epoch};
    use crate::sgd_regressor::fitting::{sk_create_state, sk_reported_parts};

    let seed = 11;
    let (features, targets) = regression_design();
    let model = SKSGDRegressor::<f64>::build()
        .alpha(0.0)
        .maximum_number_of_iterations(1)
        .random_seed(seed)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    let config = SKSGDConfig::<f64> {
        alpha: 0.0,
        learning_rate: SKSGDLearningRate::InverseScaling,
        maximum_number_of_iterations: 1,
        random_seed: seed,
        convergence_policy: SKConvergencePolicy::Permissive,
        ..SKSGDConfig::default()
    };
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    let mut state = sk_create_state(
        &config,
        SKSGDRegressorLoss::SquaredError,
        0.0,
        None,
        features.ncols(),
        targets.as_slice().unwrap(),
    )
    .unwrap();
    let order = SKRandomGenerator::new(seed).shuffled_indices(features.nrows());
    let weights = vec![1.0; features.nrows()];
    sk_run_dense_epoch(
        &mut state,
        features.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();
    let (coefficients, intercept) = sk_reported_parts(&state, features.ncols());
    assert_eq!(
        model.coefficients().as_slice().unwrap(),
        coefficients.as_slice().unwrap()
    );
    assert_eq!(model.intercept(), intercept);
}

#[test]
fn the_fit_result_is_thread_count_invariant() {
    // The seeded fit must not depend on the worker count (spec
    // `deterministic-randomness`, scenario "Thread count does not change the
    // result").
    let (features, targets) = regression_design();
    let fit_with_threads = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            SKSGDRegressor::<f64>::build()
                .random_seed(11)
                .maximum_number_of_iterations(10)
                .convergence_policy(SKConvergencePolicy::Permissive)
                .build()
                .unwrap()
                .fit(features.view(), targets.view())
                .unwrap()
        })
    };
    let single_threaded = fit_with_threads(1);
    let multi_threaded = fit_with_threads(4);
    assert_eq!(
        single_threaded.coefficients(),
        multi_threaded.coefficients()
    );
    assert_eq!(single_threaded.intercept(), multi_threaded.intercept());
}
