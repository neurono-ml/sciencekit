//! Companion TDD tests for the KKT stopping criterion and iteration tracker
//! (spec `support-vector-machines`).

use sciencekit_common::SKError;

use super::{SKSmoState, SKSmoTracker, sk_maximal_kkt_violation, sk_recover_smo_bias};

#[test]
fn smo_stopping_detects_converged_toy_problem_and_recovers_bias() {
    // 2-point shifted linear problem: x = [2.0, 4.0], y = [-1, +1], C = 10.
    // Optimal alpha = [0.5, 0.5], w = 1.0, b = -3.0.
    // G_0 = y_0*(w*x_0) - 1 = -2 - 1 = -3 => -y_0 G_0 = -3.
    // G_1 = y_1*(w*x_1) - 1 =  4 - 1 =  3 => -y_1 G_1 = -3.
    let mut state = SKSmoState::new(vec![-1, 1], vec![10.0_f64, 10.0], vec![-1.0, -1.0]);
    let row_0 = [4.0_f64, 8.0];
    let row_1 = [8.0_f64, 16.0];

    let mut tracker = SKSmoTracker::new(1e-6_f64, 10);
    let initial_violation = sk_maximal_kkt_violation(&state);
    assert!((initial_violation - 2.0).abs() < 1e-12);
    assert!(!tracker.check_or_advance(initial_violation).unwrap());
    assert_eq!(tracker.executed_iterations(), 1);

    state.update_pair(1, 0, 0.5, 0.5, &row_1, &row_0, 2);
    let final_violation = sk_maximal_kkt_violation(&state);
    assert!(final_violation <= 1e-12);
    assert!(tracker.check_or_advance(final_violation).unwrap());
    assert_eq!(tracker.executed_iterations(), 1);
    assert!((sk_recover_smo_bias(&state) - (-3.0)).abs() < 1e-12);
}

#[test]
fn smo_stopping_returns_not_converged_with_exact_iteration_count_on_exhaustion() {
    let mut tracker = SKSmoTracker::new(1e-4_f64, 3);
    assert!(!tracker.check_or_advance(1.5).unwrap());
    assert!(!tracker.check_or_advance(0.8).unwrap());
    assert!(!tracker.check_or_advance(0.2).unwrap());
    assert_eq!(tracker.executed_iterations(), 3);

    let err = tracker.check_or_advance(0.05).unwrap_err();
    assert!(matches!(err, SKError::NotConverged { iterations: 3 }));
}
