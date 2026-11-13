//! Dense lasso fit, sparsity, optimality and prediction tests (spec `lasso`).

use crate::coordinate_descent::test_datasets::{
    assert_close, assert_close_scalar, correlated_design, one_feature, orthogonal_design,
    soft_threshold, zero_mean_feature,
};
use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};

use super::builder::SKLassoBuilder;
use super::core_implementation::SKLasso;

/// A tight, deterministic lasso over `f64`.
fn lasso(alpha: f64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(alpha)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .expect("lasso builds")
}

/// A fresh default builder.
fn default_builder() -> SKLassoBuilder<f64> {
    SKLasso::build()
}

/// The default configuration is documented.
#[test]
fn default_configuration() {
    let estimator = SKLasso::<f64>::build().build().unwrap();
    assert_eq!(estimator.alphas(), &[1.0]);
    assert!(estimator.include_intercept());
    assert!(!estimator.positive());
    assert!(estimator.screening());
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.maximum_number_of_iterations(), 1000);
    assert!((estimator.tolerance() - 1e-4).abs() < 1e-12);
    assert_eq!(estimator.random_seed(), 0);
}

/// On the orthogonal design the coefficients match `S(βⱼ, α)`.
#[test]
fn orthogonal_design_matches_the_closed_form() {
    let (features, targets) = orthogonal_design();
    let model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    assert_close(&model.coefficient_for(0).to_vec(), &[1.5, -0.5, 0.0], 1e-6);
    assert_close_scalar(model.intercept_for(0), 0.0, 1e-8);
}

/// A feature the penalty excludes is driven to an exact zero.
#[test]
fn irrelevant_feature_is_exactly_zero() {
    let (features, targets) = orthogonal_design();
    let model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.coefficient_for(0)[2], 0.0);
}

/// The converged coefficients satisfy the lasso KKT conditions.
#[test]
fn kkt_conditions_hold() {
    let (features, targets) = orthogonal_design();
    let alpha = 0.5;
    let model = lasso(alpha).fit(features.view(), targets.view()).unwrap();
    let coefficients = model.coefficient_for(0);
    let intercept = model.intercept_for(0);
    let residual = &targets - &features.dot(&coefficients) - intercept;
    let gradient = features.t().dot(&residual) / features.nrows() as f64;
    for (column, &coefficient) in coefficients.iter().enumerate() {
        if coefficient.abs() > 1e-8 {
            assert!((gradient[column].abs() - alpha).abs() < 1e-6);
        } else {
            assert!(gradient[column].abs() <= alpha + 1e-6);
        }
    }
}

/// A sequence of alphas yields one solution per value, in order.
#[test]
fn alpha_sequence_yields_one_solution_per_value() {
    let (features, targets) = orthogonal_design();
    let estimator = SKLasso::build()
        .alpha(vec![0.25, 0.5, 1.5])
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.alphas(), &[0.25, 0.5, 1.5]);
    assert_eq!(model.coefficients().nrows(), 3);
    let expected = [
        soft_threshold(2.0, 0.25),
        soft_threshold(2.0, 0.5),
        soft_threshold(2.0, 1.5),
    ];
    for (row, &value) in expected.iter().enumerate() {
        assert_close_scalar(model.coefficient_for(row)[0], value, 1e-6);
    }
}

/// The penalty does not shrink the intercept.
#[test]
fn intercept_is_not_penalized() {
    let (features, targets) = zero_mean_feature();
    let weak = lasso(1e-9).fit(features.view(), targets.view()).unwrap();
    let strong = lasso(10.0).fit(features.view(), targets.view()).unwrap();
    assert!(strong.coefficient_for(0)[0].abs() < weak.coefficient_for(0)[0].abs());
    assert_close_scalar(weak.intercept_for(0), 5.0, 1e-6);
    assert_close_scalar(strong.intercept_for(0), 5.0, 1e-6);
}

/// With the intercept disabled the model reports a zero intercept.
#[test]
fn intercept_can_be_disabled() {
    let (features, targets) = one_feature();
    let estimator = SKLasso::build()
        .alpha(1.0)
        .include_intercept(false)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.intercept_for(0), 0.0);
    assert!(!model.include_intercept());
}

/// The positive constraint clamps a negative coefficient at zero.
#[test]
fn positive_constraint_clamps() {
    let (features, targets) = orthogonal_design();
    let estimator = SKLasso::build()
        .alpha(0.5)
        .positive(true)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert!(model.positive());
    assert_close(&model.coefficient_for(0).to_vec(), &[1.5, 0.0, 0.0], 1e-6);
}

/// Prediction evaluates the learned line on new rows.
#[test]
fn predicts_fitted_line() {
    let (features, targets) = one_feature();
    let model = lasso(1e-9).fit(features.view(), targets.view()).unwrap();
    let predictions = model.predict(array![[4.0]].view()).unwrap();
    assert_close_scalar(predictions[0], 8.0, 1e-4);
}

/// Plain prediction refuses a multi-alpha model.
#[test]
fn plain_predict_refuses_multiple_alphas() {
    let (features, targets) = one_feature();
    let estimator = SKLasso::build()
        .alpha(vec![0.5, 1.0])
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    let error = model.predict(features.view()).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A non-default backend reaches the fitted model.
#[test]
fn non_default_backend_reaches_model() {
    let (features, targets) = one_feature();
    let estimator = default_builder()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

/// A random-selection lasso whose sweep order is fully described by the seed.
fn random_selection_lasso(seed: u64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(0.05)
        .selection(crate::coordinate_descent::SKSelection::Random)
        .random_seed(seed)
        .maximum_number_of_iterations(100_000)
        // A loose tolerance stops before the optimum, keeping the seeded sweep
        // order observable in the coefficients.
        .tolerance(1e-2)
        .build()
        .expect("lasso builds")
}

#[test]
fn random_seed_defaults_to_zero_and_reaches_the_model() {
    let estimator = SKLasso::<f64>::build().build().unwrap();
    assert_eq!(estimator.random_seed(), 0);
    let (features, targets) = orthogonal_design();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert_eq!(model.random_seed(), 0);

    let seeded = SKLasso::<f64>::build().random_seed(9).build().unwrap();
    assert_eq!(seeded.random_seed(), 9);
}

#[test]
fn random_selection_same_seed_reproduces_the_coefficients() {
    let (features, targets) = orthogonal_design();
    let first = random_selection_lasso(7)
        .fit(features.view(), targets.view())
        .unwrap();
    let second = random_selection_lasso(7)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(first.coefficients(), second.coefficients());
    assert_eq!(first.intercepts(), second.intercepts());
    assert_eq!(first.random_seed(), 7);
}

#[test]
fn random_selection_different_seeds_can_change_the_coefficients() {
    let (features, targets) = correlated_design();
    let first = random_selection_lasso(1)
        .fit(features.view(), targets.view())
        .unwrap();
    let second = random_selection_lasso(2)
        .fit(features.view(), targets.view())
        .unwrap();
    assert_ne!(first.coefficients(), second.coefficients());
}

#[test]
fn parallel_per_alpha_results_are_stored_by_index_for_every_thread_count() {
    // Each alpha is an independent parallel component with a generator derived
    // from its configured position; the result rows must follow that index
    // whichever thread finishes first (spec scenario "Stable order").
    let (features, targets) = correlated_design();
    let fit_with_threads = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            SKLasso::build()
                .alpha(vec![0.05, 0.1, 0.2])
                .selection(crate::coordinate_descent::SKSelection::Random)
                .random_seed(13)
                .tolerance(1e-4)
                .maximum_number_of_iterations(100_000)
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
    assert_eq!(single_threaded.intercepts(), multi_threaded.intercepts());
}
