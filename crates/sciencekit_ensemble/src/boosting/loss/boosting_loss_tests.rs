//! Companion tests for regression boosting losses (tasks 2.1 and 2.2).

use super::{SKGradientLoss, SKSquaredErrorLoss};

/// Squared-error loss initializes at the target mean, produces negative
/// gradient `y - p`, constant unit Hessian `1`, identity link/inverse-link, and
/// half squared residual loss `0.5 * (y - p)²`.
#[test]
fn squared_error_loss_matches_hand_computed_contract() {
    let loss = SKSquaredErrorLoss::new();
    let targets = [1.0_f64, 3.0, 5.0, 7.0];
    assert!((loss.initial_prediction(&targets) - 4.0).abs() < 1e-12);
    assert_eq!(loss.initial_prediction(&[] as &[f64]), 0.0);

    // Scenario: Squared-error gradient -> negative gradient is y - p, Hessian is 1.
    assert!((loss.negative_gradient(5.5_f64, 2.0) - 3.5).abs() < 1e-12);
    assert!((loss.negative_gradient(-1.0_f64, 2.5) - (-3.5)).abs() < 1e-12);
    assert!((loss.hessian(5.5_f64, 2.0) - 1.0).abs() < 1e-12);
    assert!((loss.hessian(-1.0_f64, 2.5) - 1.0).abs() < 1e-12);

    assert!((loss.link(3.25_f64) - 3.25).abs() < 1e-12);
    assert!((loss.inverse_link(-4.5_f64) - (-4.5)).abs() < 1e-12);
    // 0.5 * (6.0 - 2.0)^2 = 8.0
    assert!((loss.loss(2.0_f64, 6.0) - 8.0).abs() < 1e-12);
}

/// Absolute-error loss returns signed `-1 / 0 / +1` negative gradients,
/// documented constant unit Hessian `1`, target-mean initialization and `|y-p|`.
#[test]
fn absolute_error_loss_uses_sign_gradient_and_unit_hessian() {
    use super::SKAbsoluteErrorLoss;

    let loss = SKAbsoluteErrorLoss::new();
    assert!((loss.initial_prediction(&[2.0_f64, 4.0, 6.0]) - 4.0).abs() < 1e-12);

    assert_eq!(loss.negative_gradient(5.0_f64, 2.0), 1.0);
    assert_eq!(loss.negative_gradient(2.0_f64, 5.0), -1.0);
    assert_eq!(loss.negative_gradient(3.0_f64, 3.0), 0.0);

    assert_eq!(loss.hessian(5.0_f64, 2.0), 1.0);
    assert_eq!(loss.hessian(2.0_f64, 5.0), 1.0);
    assert_eq!(loss.hessian(3.0_f64, 3.0), 1.0);

    assert!((loss.loss(2.0_f64, -1.5) - 3.5).abs() < 1e-12);
}

/// Huber loss matches squared error inside `[-delta, delta]` and clips the
/// negative gradient to `±delta` outside while keeping a positive unit Hessian.
#[test]
fn huber_loss_clips_gradient_and_transitions_smoothly_at_delta() {
    use super::SKHuberLoss;
    use sciencekit_common::SKError;

    assert!(matches!(
        SKHuberLoss::<f64>::build().delta(0.0).build(),
        Err(SKError::InvalidHyperparameter { name: "delta", .. })
    ));

    let loss = SKHuberLoss::<f64>::build()
        .delta(1.5)
        .build()
        .expect("huber builds");
    assert!((loss.initial_prediction(&[1.0, 5.0]) - 3.0).abs() < 1e-12);

    // Inside band |y - p| = 1.0 <= 1.5: unclipped residual and 0.5 * 1.0^2 = 0.5.
    assert!((loss.negative_gradient(3.0, 2.0) - 1.0).abs() < 1e-12);
    assert_eq!(loss.hessian(3.0, 2.0), 1.0);
    assert!((loss.loss(2.0, 3.0) - 0.5).abs() < 1e-12);

    // Outside band |y - p| = 4.0 > 1.5: clipped to ±1.5 and 1.5 * (4.0 - 0.75) = 4.875.
    assert!((loss.negative_gradient(6.0, 2.0) - 1.5).abs() < 1e-12);
    assert!((loss.negative_gradient(-2.0, 2.0) - (-1.5)).abs() < 1e-12);
    assert_eq!(loss.hessian(6.0, 2.0), 1.0);
    assert!((loss.loss(2.0, 6.0) - 4.875).abs() < 1e-12);
}
