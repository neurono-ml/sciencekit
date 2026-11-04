//! Companion tests for the stochastic-gradient loss catalogue
//! (spec `stochastic-gradient-solver`).

use super::definition::SKSGDLoss;
use super::regressor::{SKSGDRegressorLossAdapter, sk_regressor_loss_adapter};
use super::{SKSGDClassifierLoss, SKSGDRegressorLoss};

const TOLERANCE: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn hinge_is_zero_beyond_the_margin() {
    let loss = SKSGDClassifierLoss::Hinge;
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 0.5, 1.0),
        0.5,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.5, 1.0),
        -1.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 2.0, 1.0),
        0.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 2.0, 1.0),
        0.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 2.0, -1.0),
        3.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 2.0, -1.0),
        1.0,
    );
}

#[test]
fn log_loss_matches_the_logistic_form() {
    let loss = SKSGDClassifierLoss::LogLoss;
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 0.0, 1.0),
        2.0_f64.ln(),
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.0, 1.0),
        -0.5,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.0, -1.0),
        0.5,
    );
    assert!(<SKSGDClassifierLoss as SKSGDLoss<f64>>::is_probabilistic(
        &loss
    ));
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::probability(&loss, 0.0),
        0.5,
    );
}

#[test]
fn modified_huber_switches_to_the_linear_region_below_minus_one() {
    let loss = SKSGDClassifierLoss::ModifiedHuber;
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 0.0, 1.0),
        1.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.0, 1.0),
        -2.0,
    );
    // y·z = -2 < -1: value = -4·y·z = 8, slope = -4·y = -4.
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, -2.0, 1.0),
        8.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, -2.0, 1.0),
        -4.0,
    );
    assert!(<SKSGDClassifierLoss as SKSGDLoss<f64>>::is_probabilistic(
        &loss
    ));
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::probability(&loss, 0.0),
        0.5,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::probability(&loss, 10.0),
        1.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::probability(&loss, -10.0),
        0.0,
    );
}

#[test]
fn squared_hinge_is_the_square_of_the_violation() {
    let loss = SKSGDClassifierLoss::SquaredHinge;
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 0.0, 1.0),
        1.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.0, 1.0),
        -2.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 3.0, 1.0),
        0.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 3.0, 1.0),
        0.0,
    );
}

#[test]
fn perceptron_punishes_only_misclassified_samples() {
    let loss = SKSGDClassifierLoss::Perceptron;
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, 0.5, 1.0),
        0.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, 0.5, 1.0),
        0.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::value(&loss, -1.0, 1.0),
        1.0,
    );
    assert_close(
        <SKSGDClassifierLoss as SKSGDLoss<f64>>::slope(&loss, -1.0, 1.0),
        -1.0,
    );
}

#[test]
fn non_probabilistic_classifier_losses_report_no_probability() {
    for loss in [
        SKSGDClassifierLoss::Hinge,
        SKSGDClassifierLoss::SquaredHinge,
        SKSGDClassifierLoss::Perceptron,
    ] {
        assert!(!<SKSGDClassifierLoss as SKSGDLoss<f64>>::is_probabilistic(
            &loss
        ));
        assert!(<SKSGDClassifierLoss as SKSGDLoss<f64>>::probability(&loss, 0.0).is_nan());
    }
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

#[test]
fn the_loss_catalogue_works_on_f32() {
    let loss = SKSGDClassifierLoss::LogLoss;
    assert!(
        (<SKSGDClassifierLoss as SKSGDLoss<f32>>::value(&loss, 0.0, 1.0) - 2.0_f32.ln()).abs()
            < 1e-6
    );
    assert!((<SKSGDClassifierLoss as SKSGDLoss<f32>>::slope(&loss, 0.0, 1.0) + 0.5).abs() < 1e-6);
}

fn value(loss: &SKSGDRegressorLossAdapter<f64>, prediction: f64, target: f64) -> f64 {
    <SKSGDRegressorLossAdapter<f64> as SKSGDLoss<f64>>::value(loss, prediction, target)
}

fn slope(loss: &SKSGDRegressorLossAdapter<f64>, prediction: f64, target: f64) -> f64 {
    <SKSGDRegressorLossAdapter<f64> as SKSGDLoss<f64>>::slope(loss, prediction, target)
}
