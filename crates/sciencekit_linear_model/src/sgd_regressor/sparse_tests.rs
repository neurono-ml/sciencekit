//! Companion tests for zero-copy CSR fits of the SGD regressor (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKRegressorPredictor, SKSupervisedFit};
use sprs::TriMat;

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDPenalty, SKSGDRegressorLoss};
use crate::test_datasets::{assert_close, assert_close_f32, regression_design, to_csr};

fn fit_dense(
    features: &ndarray::Array2<f64>,
    targets: &Array1<f64>,
    loss: SKSGDRegressorLoss,
    penalty: SKSGDPenalty,
) -> super::SKSGDRegressorModel<f64> {
    SKSGDRegressor::<f64>::build()
        .loss(loss)
        .epsilon(0.5)
        .penalty(penalty)
        .alpha(0.1)
        .l1_ratio(0.5)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .maximum_number_of_iterations(20)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
}

#[test]
fn dense_and_csr_fits_agree_across_losses_and_penalties() {
    let (features, targets) = regression_design();
    let sparse = to_csr(&features);
    for loss in [
        SKSGDRegressorLoss::SquaredError,
        SKSGDRegressorLoss::Huber,
        SKSGDRegressorLoss::EpsilonInsensitive,
        SKSGDRegressorLoss::SquaredEpsilonInsensitive,
    ] {
        for penalty in [
            SKSGDPenalty::L2,
            SKSGDPenalty::L1,
            SKSGDPenalty::ElasticNet,
            SKSGDPenalty::None,
        ] {
            let dense_model = fit_dense(&features, &targets, loss, penalty);
            let sparse_model = SKSGDRegressor::<f64>::build()
                .loss(loss)
                .epsilon(0.5)
                .penalty(penalty)
                .alpha(0.1)
                .l1_ratio(0.5)
                .learning_rate(SKSGDLearningRate::Constant)
                .initial_learning_rate(0.05)
                .shuffle(false)
                .maximum_number_of_iterations(20)
                .convergence_policy(SKConvergencePolicy::Permissive)
                .build()
                .unwrap()
                .fit(sparse.view(), targets.view())
                .unwrap();
            assert_close(
                dense_model.coefficients().as_slice().unwrap(),
                sparse_model.coefficients().as_slice().unwrap(),
                1e-12,
            );
            assert_close(
                &[dense_model.intercept()],
                &[sparse_model.intercept()],
                1e-12,
            );
        }
    }
}

#[test]
fn sparse_prediction_matches_dense() {
    let (features, targets) = regression_design();
    let sparse = to_csr(&features);
    let model = fit_dense(
        &features,
        &targets,
        SKSGDRegressorLoss::SquaredError,
        SKSGDPenalty::L2,
    );
    let dense_prediction = model.predict(features.view()).unwrap();
    let sparse_prediction = model.predict(sparse.view()).unwrap();
    assert_close(
        dense_prediction.as_slice().unwrap(),
        sparse_prediction.as_slice().unwrap(),
        1e-12,
    );
}

#[test]
fn a_sparse_fit_supports_both_floats() {
    let targets = array![2.0f32, 4.0, 6.0];
    let mut triplets = TriMat::new((3, 1));
    triplets.add_triplet(0, 0, 1.0f32);
    triplets.add_triplet(1, 0, 2.0f32);
    triplets.add_triplet(2, 0, 3.0f32);
    let sparse = triplets.to_csr();
    let model = SKSGDRegressor::<f32>::build()
        .alpha(0.0)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .shuffle(false)
        .maximum_number_of_iterations(2_000)
        .tolerance(1e-6)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    let predicted = model.predict(sparse.view()).unwrap();
    assert!(predicted.iter().all(|value| value.is_finite()));
    assert_close_f32(
        predicted.as_slice().unwrap(),
        targets.as_slice().unwrap(),
        0.1,
    );
}
