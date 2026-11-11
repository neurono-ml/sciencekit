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
