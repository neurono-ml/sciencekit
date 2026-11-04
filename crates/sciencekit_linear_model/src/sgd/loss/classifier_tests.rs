//! Companion tests for the classifier loss catalogue (spec
//! `stochastic-gradient-solver`, decision 2).

use super::SKSGDClassifierLoss;
use super::definition::SKSGDLoss;

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
fn the_loss_catalogue_works_on_f32() {
    let loss = SKSGDClassifierLoss::LogLoss;
    assert!(
        (<SKSGDClassifierLoss as SKSGDLoss<f32>>::value(&loss, 0.0, 1.0) - 2.0_f32.ln()).abs()
            < 1e-6
    );
    assert!((<SKSGDClassifierLoss as SKSGDLoss<f32>>::slope(&loss, 0.0, 1.0) + 0.5).abs() < 1e-6);
}
