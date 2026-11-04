//! Companion tests for the regressor loss catalogue (spec
//! `stochastic-gradient-solver`, decision 2).

use super::SKSGDRegressorLoss;
use super::definition::SKSGDLoss;
use super::regressor::{SKSGDRegressorLossAdapter, sk_regressor_loss_adapter};

const TOLERANCE: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

fn value(loss: &SKSGDRegressorLossAdapter<f64>, prediction: f64, target: f64) -> f64 {
    <SKSGDRegressorLossAdapter<f64> as SKSGDLoss<f64>>::value(loss, prediction, target)
}

fn slope(loss: &SKSGDRegressorLossAdapter<f64>, prediction: f64, target: f64) -> f64 {
    <SKSGDRegressorLossAdapter<f64> as SKSGDLoss<f64>>::slope(loss, prediction, target)
}

#[test]
fn squared_error_is_half_the_squared_residual() {
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 1.0);
    assert_close(value(&loss, 3.0, 1.0), 2.0);
    assert_close(slope(&loss, 3.0, 1.0), 2.0);
    assert_close(value(&loss, 1.0, 1.0), 0.0);
    assert_close(slope(&loss, 1.0, 1.0), 0.0);
}

#[test]
fn huber_is_quadratic_inside_the_band_and_linear_outside_it() {
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::Huber, 1.0);
    // Residual 0.5 inside the band: 0.5·0.5² = 0.125, slope 0.5.
    assert_close(value(&loss, 0.5, 0.0), 0.125);
    assert_close(slope(&loss, 0.5, 0.0), 0.5);
    // Residual 2 outside the band: 1·(2 − 0.5) = 1.5, slope ±1.
    assert_close(value(&loss, 2.0, 0.0), 1.5);
    assert_close(slope(&loss, 2.0, 0.0), 1.0);
    assert_close(slope(&loss, -2.0, 0.0), -1.0);
}

#[test]
fn insensitive_losses_ignore_residuals_inside_the_band() {
    let insensitive = sk_regressor_loss_adapter(SKSGDRegressorLoss::EpsilonInsensitive, 0.5);
    assert_close(value(&insensitive, 0.3, 0.0), 0.0);
    assert_close(slope(&insensitive, 0.3, 0.0), 0.0);
    assert_close(value(&insensitive, 1.5, 0.0), 1.0);
    assert_close(slope(&insensitive, 1.5, 0.0), 1.0);
    assert_close(slope(&insensitive, -1.5, 0.0), -1.0);

    let squared = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredEpsilonInsensitive, 0.5);
    assert_close(value(&squared, 0.3, 0.0), 0.0);
    assert_close(value(&squared, 1.5, 0.0), 0.5);
    assert_close(slope(&squared, 1.5, 0.0), 1.0);
}

#[test]
fn every_regressor_loss_is_non_probabilistic() {
    for loss in [
        SKSGDRegressorLoss::SquaredError,
        SKSGDRegressorLoss::Huber,
        SKSGDRegressorLoss::EpsilonInsensitive,
        SKSGDRegressorLoss::SquaredEpsilonInsensitive,
    ] {
        let adapter = sk_regressor_loss_adapter(loss, 1.0);
        assert!(!<SKSGDRegressorLossAdapter<f64> as SKSGDLoss<f64>>::is_probabilistic(&adapter));
    }
}
